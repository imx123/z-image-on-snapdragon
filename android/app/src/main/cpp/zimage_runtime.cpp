// Z-Image on-device runtime — QNN segmented inference.
// Mixed-backend design:
//   - Transformer DiT     -> r.main  (backend.txt: htp|gpu|cpu; production = gpu v10 fp32)
//   - VAE decoder + Qwen3  -> r.hp   (fixed HTP FP16)
#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <thread>
#include <unordered_map>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include "QnnBackend.h"
#include "GPU/QnnGpuBackend.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"
#include "QnnTensor.h"
#include "QnnGraph.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnLog.h"
#include "QnnInterface.h"

struct GraphInfo{Qnn_GraphHandle_t graph;char* graphName;Qnn_Tensor_t* inputTensors;uint32_t numInputTensors;Qnn_Tensor_t* outputTensors;uint32_t numOutputTensors;};
struct GraphConfigInfo{char* graphName;const QnnGraph_Config_t** graphConfigs;};
typedef int32_t ModelError; // QnnModel_composeGraphs returns qnn_wrapper_api::ModelError_t (enum; 0 = OK)
using ComposeFn=ModelError(*)(Qnn_BackendHandle_t,QNN_INTERFACE_VER_TYPE,Qnn_ContextHandle_t,const GraphConfigInfo**,uint32_t,GraphInfo***,uint32_t*,bool,QnnLog_Callback_t,QnnLog_Level_t);
using FreeFn=ModelError(*)(GraphInfo***,uint32_t); using ProvidersFn=Qnn_ErrorHandle_t(*)(const QnnInterface_t***,uint32_t*);
struct Runtime;
struct SegmentGraph{std::string name;std::string libName;void* lib=nullptr;ComposeFn compose=nullptr;FreeFn freeGraphs=nullptr;GraphInfo** graphs=nullptr;uint32_t count=0;std::string error;
  // context-binary mmap lifetime (must outlive createFromBinary's lazy reads)
  int binFd=-1; void* binAddr=nullptr; size_t binSize=0;
  void unmapBin(){ 
    if(binAddr&&binSize){ madvise(binAddr,binSize,MADV_DONTNEED); munmap(binAddr,binSize); }
    if(binFd>=0) close(binFd); binAddr=nullptr; binSize=0; binFd=-1; }
};
struct QnnSet{
  std::string label="?";
  void* lib=nullptr;
  QNN_INTERFACE_VER_TYPE api{};
  Qnn_LogHandle_t log=nullptr;
  Qnn_BackendHandle_t backend=nullptr;
  Qnn_DeviceHandle_t device=nullptr;
  Qnn_ContextHandle_t context=nullptr;
  bool alias=false;
  void release(Runtime* r);
};
struct Runtime{uint32_t magic=0x5A494D47;bool tcacheFrontendFp32=true;std::string root;void* sys=nullptr;void* gpu=nullptr;void* htp=nullptr;void* vae=nullptr;void* model=nullptr;std::string modelName;int backendKind=0; // 0=HTP, 1=CPU, 2=GPU (transformer / main)
QnnSet main;   // transformer backend (backend.txt)
QnnSet hp;     // fixed HTP backend for VAE + text encoder
Qnn_LogHandle_t logHandle=nullptr;QNN_INTERFACE_VER_TYPE api{};Qnn_BackendHandle_t backend=nullptr;Qnn_DeviceHandle_t device=nullptr;Qnn_ContextHandle_t context=nullptr;GraphInfo** graphs=nullptr;uint32_t count=0;ComposeFn compose=nullptr;FreeFn freeGraphs=nullptr;std::vector<SegmentGraph> segments;bool transformerLoaded=false;std::string transformerError;std::vector<SegmentGraph> textSegments;bool textLoaded=false;std::string textError;std::string error;std::string info;
// P0 cache: transformer segments composed once, reused across the 8 denoise
// steps. Order: [0]=frontend, [1..10]=layer groups (fp32) or 5 (fp16), last=final.
std::vector<SegmentGraph> tcache; bool tcacheReady=false; std::string tcacheError;
std::vector<Qnn_ContextHandle_t> tcCtx; // per-segment contexts, freed in nativeDestroy
bool tcacheStepRestore=false; int tcacheGroupCount=10; // LMK workaround state
// Text-encoder cache (same idea): [0]=embedding, [1..6]=layer groups.
std::vector<SegmentGraph> textCache; bool textCacheReady=false; std::string textCacheError;
std::vector<Qnn_ContextHandle_t> textCtx;
// Cached frontend outputs that stay constant across steps (cap_feats/mask only).
std::vector<uint8_t> tcMaskOut, tcAdalnOut;
};
void logline(Runtime&r,const std::string&s){std::string p=r.root+"/jni.log";FILE*f=fopen(p.c_str(),"a");if(f){fprintf(f,"%s\n",s.c_str());fclose(f);}__android_log_print(ANDROID_LOG_INFO,"zimage-jni","%s",s.c_str());}
void writeProgress(Runtime&r,const std::string&stage,int pct,int total){
  std::string p=r.root+"/progress.txt";
  FILE*f=fopen(p.c_str(),"w");
  if(f){ fprintf(f,"%s\n%d\n%d\n",stage.c_str(),pct,total); fclose(f); }
}
static void* g_sysLib=nullptr;      // set in init(); libQnnSystem.so handle
extern void* qnnSystemLib();
void* qnnSystemLib(){ return g_sysLib; }
void qnnLogCb(const char* fmt,QnnLog_Level_t level,uint64_t ts,va_list ap){if(!fmt)return;char buf[2048];vsnprintf(buf,sizeof(buf),fmt,ap);FILE*f=fopen("/data/user/0/com.example.zimage/files/zimage-runtime/jni.log","a");if(f){fprintf(f,"[QNN] %s\n",buf);fclose(f);}__android_log_print(ANDROID_LOG_INFO,"zimage-qnn","%s",buf);}
Runtime* rt(jlong h){return reinterpret_cast<Runtime*>(h);}
void* load(const std::string&p,int flags=RTLD_NOW|RTLD_LOCAL){return dlopen(p.c_str(),flags);}
void* bareLoad(Runtime&r,const std::string&name,int flags){
  logline(r,"bare-dlopen-begin "+name);
  ::dlerror();
  void*h=dlopen(name.c_str(),flags);
  std::string msg="bare-dlopen-end "+name+" -> ";
  if(h){msg+="ok";}
  else{const char*e=dlerror();msg+=(e?e:"<null-dlerror>");}
  logline(r,msg);
  return h;
}
std::string err(Qnn_ErrorHandle_t x){return std::to_string(static_cast<uint32_t>(x));}
bool select(void*h,QNN_INTERFACE_VER_TYPE& out){auto fn=reinterpret_cast<ProvidersFn>(dlsym(h,"QnnInterface_getProviders"));if(!fn)return false;const QnnInterface_t** p=nullptr;uint32_t n=0;if(fn(&p,&n)!=QNN_SUCCESS||!p)return false;for(uint32_t i=0;i<n;i++){auto v=p[i]->apiVersion.coreApiVersion;if(v.major==QNN_API_VERSION_MAJOR&&v.minor>=QNN_API_VERSION_MINOR){out=p[i]->QNN_INTERFACE_VER_NAME;return true;}}return false;}
size_t bytes(const Qnn_Tensor_t&t){size_t n=1;for(uint32_t i=0;i<t.v1.rank;i++)n*=t.v1.dimensions[i];size_t b=4;if(t.v1.dataType==QNN_DATATYPE_FLOAT_16)b=2;else if(t.v1.dataType==QNN_DATATYPE_UINT_8||t.v1.dataType==QNN_DATATYPE_INT_8||t.v1.dataType==QNN_DATATYPE_BOOL_8)b=1;else if(t.v1.dataType==QNN_DATATYPE_FLOAT_64||t.v1.dataType==QNN_DATATYPE_INT_64||t.v1.dataType==QNN_DATATYPE_UINT_64)b=8;return n*b;}
std::string ti(const Qnn_Tensor_t&t){std::ostringstream o;o<<(t.v1.name?t.v1.name:"?")<<"[";for(uint32_t i=0;i<t.v1.rank;i++){if(i)o<<",";o<<t.v1.dimensions[i];}o<<"]";return o.str();}
const char* backendLabel(int kind){return kind==2?"gpu":(kind==1?"cpu":"htp");}
void QnnSet::release(Runtime* r){
  if(alias) return;
  if(context&&api.contextFree) api.contextFree(context,nullptr);
  if(device&&api.deviceFree) api.deviceFree(device);
  if(backend&&api.backendFree) api.backendFree(backend);
  if(log&&api.logFree) api.logFree(log);
  if(lib) dlclose(lib);
}

bool makeBackend(Runtime& r, QnnSet& s, const char* libName, bool htpPreloads){
  auto p=r.root+"/lib/arm64-v8a/";
  s.lib=load(p+libName); // RTLD_LOCAL: two backends must not interpose each other
  logline(r,s.label+" backend lib "+libName+": "+(s.lib?"ok":"fail"));
  if(!s.lib||!select(s.lib,s.api)){ r.error=s.label+" interface unavailable"; logline(r,"ERROR "+r.error); return false; }
  if(!s.api.backendCreate||!s.api.contextCreate||!s.api.graphExecute){ r.error=s.label+" API incomplete"; logline(r,"ERROR "+r.error); return false; }
  if(s.api.logCreate){ auto lx=s.api.logCreate(qnnLogCb,QNN_LOG_LEVEL_INFO,&s.log); logline(r,s.label+" logCreate="+std::to_string(static_cast<uint32_t>(lx))); }
  // NOTE: QNN_GPU_BACKEND_CONFIG_OPTION_WEIGHT_SHARING_ENABLED crashed the
  // Adreno 830 driver inside backendCreate (SIGSEGV in libQnnGpu.so) — do not
  // use backend-level custom config here. Memory experiment moved to
  // graph-level disableMemoryOptimizations in composeSegment.
  auto x=s.api.backendCreate(s.log,nullptr,&s.backend); logline(r,s.label+" backendCreate="+err(x));
  if(x!=QNN_SUCCESS){ r.error=s.label+" backendCreate="+err(x); return false; }
  if(htpPreloads){
    auto libDir=r.root+"/lib/arm64-v8a";
    ::setenv("ADSP_LIBRARY_PATH",libDir.c_str(),1);
    logline(r,"ADSP_LIBRARY_PATH="+libDir);
    bareLoad(r,"libcdsprpc.so",RTLD_NOW|RTLD_GLOBAL);
    bareLoad(r,"libQnnHtpV79Stub.so",RTLD_NOW|RTLD_GLOBAL);
    logline(r,"preloads-done, before deviceCreate");
  }
  const QnnDevice_Config_t* devCfgPtr[1]={nullptr};
  if(s.api.deviceCreate){ x=s.api.deviceCreate(s.log,devCfgPtr,&s.device); logline(r,s.label+" deviceCreate="+err(x)+" dev="+(s.device?"set":"null")); if(x!=QNN_SUCCESS){ s.device=nullptr; } }
  x=s.api.contextCreate(s.backend,s.device,nullptr,&s.context); logline(r,s.label+" contextCreate="+err(x));
  if(x!=QNN_SUCCESS){ r.error=s.label+" contextCreate="+err(x); return false; }
  return true;
}

bool composeVae(Runtime&r, QnnSet& s){
  if(r.backendKind!=0 && s.alias){
    // Non-HTP transformer AND no dedicated HTP set: GPU/CPU VAE is not viable
    // (fanout NaN on GPU). Keep r.graphs empty so nativeGenerate reports
    // VAE unavailable instead of hanging.
    r.modelName="libqnn_vae_shiftfix.so (skipped on "+s.label+")";
    logline(r,"VAE skipped on "+s.label+"; only composed on HTP");
    return true;
  }
  auto vaeLib=r.root+"/lib/arm64-v8a/libqnn_vae_shiftfix.so";
  r.model=load(vaeLib);
  r.modelName="libqnn_vae_shiftfix.so";
  if(!r.model){
    const char* dlErr=dlerror();
    logline(r,std::string("shiftfix VAE dlopen failed: ")+(dlErr?dlErr:"<null>"));
    r.model=load(r.root+"/lib/arm64-v8a/libqnn_vae_gpu.so");
    r.modelName="libqnn_vae_gpu.so";
  }
  if(!r.model){r.error="model dlopen failed";return false;}
  logline(r,"VAE model lib: "+r.modelName+" on "+s.label);r.compose=reinterpret_cast<ComposeFn>(dlsym(r.model,"QnnModel_composeGraphs"));r.freeGraphs=reinterpret_cast<FreeFn>(dlsym(r.model,"QnnModel_freeGraphsInfo"));if(!r.compose||!r.freeGraphs){r.error="model symbols missing";return false;}auto x=r.compose(s.backend,s.api,s.context,nullptr,0,&r.graphs,&r.count,false,nullptr,QNN_LOG_LEVEL_ERROR);if(x!=0||!r.graphs||!r.count){r.error="compose="+std::to_string(x);logline(r,"VAE compose failed: "+r.error);return false;}
  {std::ostringstream d; d<<"VAE graphPtr="<<(r.graphs[0]->graph?std::to_string((uintptr_t)r.graphs[0]->graph):"null")<<" name="<<(r.graphs[0]->graphName?r.graphs[0]->graphName:"null")<<" finalizeFn="<<(uintptr_t)s.api.graphFinalize; logline(r,d.str());}
  if(!r.graphs[0]->graph&&s.api.graphRetrieve(s.context,r.graphs[0]->graphName,&r.graphs[0]->graph)!=QNN_SUCCESS){r.error="graphRetrieve failed";return false;}
  auto fx=s.api.graphFinalize(r.graphs[0]->graph,nullptr,nullptr);logline(r,"graphFinalize="+err(fx));if(fx!=QNN_SUCCESS){r.error="graphFinalize="+err(fx);return false;}
  std::ostringstream o;o<<"graph="<<(r.graphs[0]->graphName?r.graphs[0]->graphName:"?")<<" input="<<(r.graphs[0]->numInputTensors?ti(r.graphs[0]->inputTensors[0]):"none")<<" output="<<(r.graphs[0]->numOutputTensors?ti(r.graphs[0]->outputTensors[0]):"none");r.info=o.str();return true;
}
void openClProbe(Runtime& r){
  void* t=dlopen("libOpenCL.so",RTLD_NOW|RTLD_LOCAL);
  logline(r,std::string("OpenCL probe: dlopen libOpenCL.so ")+(t?"ok":(dlerror()?dlerror():"<null>")));
  if(t){
    typedef int(*ClPlatFn)(unsigned,void*,unsigned*);
    auto fn=reinterpret_cast<ClPlatFn>(dlsym(t,"clGetPlatformIDs"));
    if(fn){unsigned n=0;int rc=fn(0,nullptr,&n);logline(r,"OpenCL probe: clGetPlatformIDs rc="+std::to_string(rc)+" num="+std::to_string(n));}
    else{logline(r,"OpenCL probe: clGetPlatformIDs symbol missing");}
    dlclose(t);
  }
}
bool init(Runtime& r){
  auto p=r.root+"/lib/arm64-v8a/";
  r.sys=load(p+"libQnnSystem.so");
  g_sysLib=r.sys;
  // Status-only preloads: keep them RTLD_LOCAL. GLOBAL here lets the two
  // backends interpose each other's identically-named exports, which breaks
  // graph compose on the second backend (MODEL_GRAPH_ERROR=4).
  r.gpu=load(p+"libQnnGpu.so");
  r.htp=load(p+"libQnnHtp.so");
  logline(r,std::string("sys=")+(r.sys?"ok":"fail")+" gpu="+(r.gpu?"ok":"fail")+" htp="+(r.htp?"ok":"fail"));
  // Dual-backend experiment (GraphInfo ABI fixed 2026-08-19): when the
  // transformer runs on GPU/CPU, give VAE+text their own HTP backend. The HTP
  // set is created FIRST — earlier GPU-first ordering left the second
  // backend's graphs failing to compose (MODEL_GRAPH_ERROR=4).
  if(r.backendKind==0){
    if(!makeBackend(r,r.main,"libQnnHtp.so",true)) return false;
    r.hp=r.main; r.hp.alias=true;
    logline(r,"NOTE: HTP-only mode; VAE/text on HTP");
  }else{
    r.hp.label="HTP2";
    bool hpOk=makeBackend(r,r.hp,"libQnnHtp.so",true); // HTP first
    if(r.backendKind==2){
      r.main.label="GPU"; openClProbe(r);
      if(!makeBackend(r,r.main,"libQnnGpu.so",false)) return false;
    }else{
      r.main.label="CPU";
      if(!makeBackend(r,r.main,"libQnnCpu.so",false)) return false;
    }
    if(!hpOk){
      r.error.clear();
      r.hp=r.main; r.hp.alias=true;
      logline(r,"NOTE: degraded to single-backend; VAE/text on "+r.hp.label);
    }else{
      logline(r,"NOTE: dual-backend mode; VAE/text on dedicated HTP");
    }
  }
  r.api=r.main.api; r.backend=r.main.backend; r.device=r.main.device; r.context=r.main.context; r.logHandle=r.main.log;
  return true;
}
uint16_t f32h(float f){
  const uint32_t u=*reinterpret_cast<const uint32_t*>(&f);
  const uint32_t sign=(u>>16)&0x8000u;
  const int32_t exp=static_cast<int32_t>(((u>>23)&0xFF))-127+15;
  const uint32_t mant=u&0x7FFFFFu;
  if(((u>>23)&0xFF)==0xFF){ if(mant) return static_cast<uint16_t>(sign|0x7E00u); return static_cast<uint16_t>(sign|0x7C00u); }
  if(exp>=31){ return static_cast<uint16_t>(sign|0x7C00u); }
  if(exp<=0){ return static_cast<uint16_t>(sign); }
  return static_cast<uint16_t>(sign|((uint32_t)exp<<10)|(mant>>13));
}
float h32(uint16_t h){
  const uint32_t sign=((uint32_t)(h&0x8000u))<<16;
  const int32_t exp=(h>>10)&0x1F;
  uint32_t mant=h&0x3FFu;
  uint32_t u;
  if(exp==0){
    if(mant==0){ u=sign; }
    else{ int e=1; while(!(mant&0x400u)){ mant<<=1; ++e; }
      u=sign|((uint32_t)(127-15+1-e)<<23)|((mant&0x3FFu)<<13); }
  } else if(exp==0x1F){
    u=sign|0x7F800000u|(mant<<13);
  } else {
    u=sign|((uint32_t)(exp+112)<<23)|(mant<<13);
  }
  float f; std::memcpy(&f,&u,4); return f;
}
bool composeSegment(Runtime& r, SegmentGraph& sg, QnnSet& s, Qnn_ContextHandle_t segCtx = nullptr){
  auto ctx = segCtx ? segCtx : r.context;
  auto path=r.root+"/lib/arm64-v8a/"+sg.libName;
  sg.lib=load(path);
  if(!sg.lib){ sg.error="dlopen failed: "+path; logline(r,"segment "+sg.name+" "+sg.error); return false; }
  sg.compose=reinterpret_cast<ComposeFn>(dlsym(sg.lib,"QnnModel_composeGraphs"));
  sg.freeGraphs=reinterpret_cast<FreeFn>(dlsym(sg.lib,"QnnModel_freeGraphsInfo"));
  if(!sg.compose||!sg.freeGraphs){ sg.error="segment symbols missing"; logline(r,"segment "+sg.name+" "+sg.error); return false; }
  auto x=sg.compose(s.backend,s.api,ctx,nullptr,0,&sg.graphs,&sg.count,false,nullptr,QNN_LOG_LEVEL_ERROR);
      if(x!=0||!sg.graphs||!sg.count){ sg.error="compose="+std::to_string(x); logline(r,"segment "+sg.name+" "+sg.error); return false; }
  {
    std::ostringstream d; d<<"segment "<<sg.name<<" graphPtr="<<(sg.graphs[0]->graph?std::to_string((uintptr_t)sg.graphs[0]->graph):"null")<<" name="<<(sg.graphs[0]->graphName?sg.graphs[0]->graphName:"null")<<" finalizeFn="<<(uintptr_t)s.api.graphFinalize;
    logline(r,d.str());
  }
  if(!sg.graphs[0]->graph && s.api.graphRetrieve(ctx,sg.graphs[0]->graphName,&sg.graphs[0]->graph)!=QNN_SUCCESS){
    sg.error="graphRetrieve failed"; logline(r,"segment "+sg.name+" "+sg.error); return false;
  }
  auto fx=s.api.graphFinalize(sg.graphs[0]->graph,nullptr,nullptr);
  if(fx!=QNN_SUCCESS){ sg.error="graphFinalize="+err(fx); logline(r,"segment "+sg.name+" "+sg.error); return false; }
  std::ostringstream o; o<<"segment "<<sg.name<<" graph="<<(sg.graphs[0]->graphName?sg.graphs[0]->graphName:"?");
  for(uint32_t i=0;i<sg.graphs[0]->numInputTensors;i++) o<<" in["<<i<<"]="<<ti(sg.graphs[0]->inputTensors[i]);
  for(uint32_t i=0;i<sg.graphs[0]->numOutputTensors;i++) o<<" out["<<i<<"]="<<ti(sg.graphs[0]->outputTensors[i]);
  logline(r,o.str());
  return true;
}

Qnn_Tensor_t* inTensor(GraphInfo& g,const char* name){
  if(!g.inputTensors) return nullptr;
  for(uint32_t i=0;i<g.numInputTensors;i++) if(g.inputTensors[i].v1.name&&std::strcmp(g.inputTensors[i].v1.name,name)==0) return &g.inputTensors[i];
  return nullptr;
}
Qnn_Tensor_t* outTensor(GraphInfo& g,const char* name){
  if(!g.outputTensors) return nullptr;
  for(uint32_t i=0;i<g.numOutputTensors;i++) if(g.outputTensors[i].v1.name&&std::strcmp(g.outputTensors[i].v1.name,name)==0) return &g.outputTensors[i];
  return nullptr;
}
struct TensorFeed{ Qnn_Tensor_t* t; void* data; size_t size; };
bool runGraph(Runtime& r, QnnSet& s, GraphInfo& g, const std::vector<TensorFeed>& ins, const std::vector<TensorFeed>& outs, int64_t* elapsedMs=nullptr){
  std::vector<Qnn_Tensor_t> inT; inT.reserve(ins.size());
  for(auto& f:ins){ if(!f.t){r.error="missing input tensor";return false;} f.t->v1.memType=QNN_TENSORMEMTYPE_RAW; f.t->v1.clientBuf.data=static_cast<uint8_t*>(f.data); f.t->v1.clientBuf.dataSize=f.size; inT.push_back(*f.t); }
  std::vector<Qnn_Tensor_t> outT; outT.reserve(outs.size());
  for(auto& f:outs){ if(!f.t){r.error="missing output tensor";return false;} f.t->v1.memType=QNN_TENSORMEMTYPE_RAW; f.t->v1.clientBuf.data=static_cast<uint8_t*>(f.data); f.t->v1.clientBuf.dataSize=f.size; outT.push_back(*f.t); }
  auto t0=std::chrono::steady_clock::now();
  auto x=s.api.graphExecute(g.graph,inT.data(),static_cast<uint32_t>(inT.size()),outT.data(),static_cast<uint32_t>(outT.size()),nullptr,nullptr);
  if(elapsedMs){ *elapsedMs=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count(); }
  if(x!=QNN_SUCCESS){ r.error="execute="+err(x); return false; }
  return true;
}
bool readFile(const std::string& path, std::vector<uint8_t>& out){
  FILE* f=fopen(path.c_str(),"rb"); if(!f) return false;
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  if(n<=0){fclose(f);return false;}
  out.resize(static_cast<size_t>(n));
  size_t got=fread(out.data(),1,out.size(),f); fclose(f);
  return got==out.size();
}
void fillFp16FromF32(const float* src, size_t n, std::vector<uint8_t>& dst, size_t dstOffsetBytes){
  auto* d=reinterpret_cast<uint16_t*>(dst.data()+dstOffsetBytes);
  for(size_t i=0;i<n;i++) d[i]=f32h(src[i]);
}
std::string fp16Stats(const std::vector<uint8_t>& buf){
  const auto* v=reinterpret_cast<const uint16_t*>(buf.data());
  size_t n=buf.size()/2;
  if(!n) return "empty";
  float mn=h32(v[0]), mx=mn, sum=0.0f;
  for(size_t i=0;i<n;i++){ float x=h32(v[i]); mn=std::min(mn,x); mx=std::max(mx,x); sum+=x; }
  std::ostringstream o; o<<"fp16["<<n<<"] min="<<mn<<" max="<<mx<<" mean="<<(sum/(float)n);
  return o.str();
}
void releaseSegment(Runtime& r, SegmentGraph& sg);
size_t dtypeBytes(Qnn_DataType_t dt){
  switch(dt){
    case QNN_DATATYPE_FLOAT_16: return 2;
    case QNN_DATATYPE_FLOAT_32: return 4;
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_BOOL_8: return 1;
    case QNN_DATATYPE_FLOAT_64:
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64: return 8;
    default: return 4;
  }
}
float elemToF32(const std::vector<uint8_t>& buf,Qnn_DataType_t dt,size_t i){
  if(dt==QNN_DATATYPE_FLOAT_16) return h32(reinterpret_cast<const uint16_t*>(buf.data())[i]);
  if(dt==QNN_DATATYPE_FLOAT_32) return reinterpret_cast<const float*>(buf.data())[i];
  return 0.0f;
}
void writeElem(std::vector<uint8_t>& buf,Qnn_DataType_t dt,size_t i,float v){
  if(dt==QNN_DATATYPE_FLOAT_16) reinterpret_cast<uint16_t*>(buf.data())[i]=f32h(v);
  else if(dt==QNN_DATATYPE_FLOAT_32) reinterpret_cast<float*>(buf.data())[i]=v;
}
void convertBuffer(const std::vector<uint8_t>& src,Qnn_DataType_t sdt,std::vector<uint8_t>& dst,Qnn_DataType_t ddt){
  size_t n=src.size()/dtypeBytes(sdt);
  dst.resize(n*dtypeBytes(ddt));
  for(size_t i=0;i<n;i++) writeElem(dst,ddt,i,elemToF32(src,sdt,i));
}
void packFloatBuffer(const float* src,size_t n,Qnn_DataType_t dt,std::vector<uint8_t>& dst){
  dst.resize(n*dtypeBytes(dt));
  if(dt==QNN_DATATYPE_FLOAT_16){ auto* d=reinterpret_cast<uint16_t*>(dst.data()); for(size_t i=0;i<n;i++) d[i]=f32h(src[i]); }
  else if(dt==QNN_DATATYPE_FLOAT_32){ std::memcpy(dst.data(),src,n*4); }
}
void packLatentBuffer(const float* srcCHW,Qnn_DataType_t dt,std::vector<uint8_t>& dst){
  const size_t C=16,H=64,W=64;
  dst.resize(C*H*W*dtypeBytes(dt));
  for(size_t hh=0;hh<H;hh++) for(size_t ww=0;ww<W;ww++) for(size_t cc=0;cc<C;cc++)
    writeElem(dst,dt,(hh*W+ww)*C+cc,srcCHW[(cc*H+hh)*W+ww]);
}
void transposeUnified(const std::vector<uint8_t>& src,Qnn_DataType_t dt,std::vector<uint8_t>& dst){
  const size_t S=1536,H=3840;
  dst.resize(S*H*dtypeBytes(dt));
  for(size_t h=0;h<H;h++) for(size_t seq=0;seq<S;seq++)
    writeElem(dst,dt,h*S+seq,elemToF32(src,dt,seq*H+h));
}
void transposeUnifiedBack(const std::vector<uint8_t>& src,Qnn_DataType_t dt,std::vector<uint8_t>& dst){
  const size_t S=1536,H=3840;
  dst.resize(S*H*dtypeBytes(dt));
  for(size_t seq=0;seq<S;seq++) for(size_t h=0;h<H;h++)
    writeElem(dst,dt,seq*H+h,elemToF32(src,dt,h*S+seq));
}
std::string tensorStatsRaw(const Qnn_Tensor_t& t,const std::vector<uint8_t>& buf){
  if(t.v1.dataType==QNN_DATATYPE_FLOAT_16) return fp16Stats(buf);
  if(t.v1.dataType==QNN_DATATYPE_FLOAT_32){
    const auto* v=reinterpret_cast<const float*>(buf.data()); size_t n=buf.size()/4;
    if(!n) return "empty";
    float mn=v[0],mx=mn; double sum=0.0;
    for(size_t i=0;i<n;i++){float x=v[i];mn=std::min(mn,x);mx=std::max(mx,x);sum+=x;}
    std::ostringstream o;o<<"fp32["<<n<<"] min="<<mn<<" max="<<mx<<" mean="<<(float)(sum/(double)n);
    return o.str();
  }
  return "dtype="+std::to_string(static_cast<uint32_t>(t.v1.dataType));
}
static std::string readSmaps(){
  FILE* f=fopen("/proc/self/smaps_rollup","r");
  if(!f) return "unavail";
  char b[128]; long long anon=-1,fileb=-1,shmem=-1;
  while(fgets(b,sizeof b,f)){
    long long v=0;
    if(sscanf(b,"Pss_Anon: %lld",&v)==1) anon=v;
    else if(sscanf(b,"Pss_File: %lld",&v)==1) fileb=v;
    else if(sscanf(b,"Pss_Shmem: %lld",&v)==1) shmem=v;
  }
  fclose(f);
  std::ostringstream o;
  o<<"anon="<<(anon/1024)<<"MB file="<<(fileb/1024)<<"MB shmem="<<(shmem/1024)<<"MB";
  return o.str();
}
static long readSelfRssMB(){
  FILE* f=fopen("/proc/self/status","r");
  if(!f) return -1;
  char b[128]; long kb=-1;
  while(fgets(b,sizeof b,f)) if(sscanf(b,"VmRSS: %ld kB",&kb)==1) break;
  fclose(f);
  return kb/1024;
}
static long readSelfRssMB();
void releaseSegment(Runtime& r, SegmentGraph& sg){
  sg.unmapBin(); // context freed by caller; mmap no longer needed
  // For compose-created graphs the model lib owns GraphInfo memory and
  // freeGraphs is mandatory. For BINARY-restored segments we allocated the
  // GraphInfo ourselves (new GraphInfo* / new GraphInfo) and the graph handle
  // belongs to the QNN context — calling freeGraphs on it aborts in free()
  // (tombstone 2026-08-24). Flag: sg.count==1 && sg.freeGraphs==nullptr.
  if(sg.graphs&&sg.freeGraphs){
    sg.freeGraphs(&sg.graphs,sg.count);
  } else if(sg.graphs){
    delete sg.graphs[0];
    delete sg.graphs;
  }
  sg.graphs=nullptr; sg.count=0;
  if(sg.lib){ dlclose(sg.lib); sg.lib=nullptr; }
  logline(r,"rss after release: "+std::to_string(readSelfRssMB())+"MB ["+readSmaps()+"]");
}
std::string f32Stats(const std::vector<uint8_t>& buf){
  const auto* v=reinterpret_cast<const float*>(buf.data());
  size_t n=buf.size()/4;
  if(!n) return "empty";
  float mn=v[0], mx=mn; double sum=0.0;
  for(size_t i=0;i<n;i++){ mn=std::min(mn,v[i]); mx=std::max(mx,v[i]); sum+=v[i]; }
  std::ostringstream o; o<<"f32["<<n<<"] min="<<mn<<" max="<<mx<<" mean="<<(float)(sum/(double)n);
  return o.str();
}

// ---------------------------------------------------------------------------
// Transformer probe (uses r.main backend — GPU v10 fp32 in production)
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jstring JNICALL Java_com_example_zimage_ZImageRuntime_nativeTransformerProbe
  (JNIEnv* e, jobject, jlong h, jfloatArray jLatent, jfloat jTimestep, jfloatArray jCapFeats, jbooleanArray jCapMask){
  auto* r=rt(h);
  QnnSet& S=r->main;
  if(!(S.backend&&S.context)){ logline(*r,"transformer probe unavailable: main backend not initialized"); return e->NewStringUTF("transformer unavailable: main backend not initialized"); }
  jsize nLatent=e->GetArrayLength(jLatent), nCap=e->GetArrayLength(jCapFeats), nMask=e->GetArrayLength(jCapMask);
  if(nLatent!=1*16*64*64||nCap!=512*2560||nMask!=512) return e->NewStringUTF("transformer probe input size mismatch");
  jfloat* latentF=e->GetFloatArrayElements(jLatent,nullptr);
  jfloat* capF=e->GetFloatArrayElements(jCapFeats,nullptr);
  jboolean* maskB=e->GetBooleanArrayElements(jCapMask,nullptr);
  if(!latentF||!capF||!maskB){ if(latentF)e->ReleaseFloatArrayElements(jLatent,latentF,JNI_ABORT); if(capF)e->ReleaseFloatArrayElements(jCapFeats,capF,JNI_ABORT); if(maskB)e->ReleaseBooleanArrayElements(jCapMask,maskB,JNI_ABORT); return e->NewStringUTF("transformer probe array access failed"); }

  std::ostringstream report;
  const size_t freqsElems=1*128*1536;

  SegmentGraph front; front.name="frontend"; front.libName="libqnn_transformer_frontend.so";
  writeProgress(*r,"transformer: frontend 编译中",1,14);
  Qnn_ContextHandle_t segCtx=nullptr;
  if(S.api.contextCreate(S.backend,S.device,nullptr,&segCtx)!=QNN_SUCCESS){ logline(*r,"frontend contextCreate failed"); return e->NewStringUTF("frontend contextCreate failed"); }
  if(!composeSegment(*r,front,S,segCtx)){ releaseSegment(*r,front); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,"frontend compose failed: "+front.error); return e->NewStringUTF(("frontend compose failed: "+front.error).c_str()); }
  auto& fg=*front.graphs[0];
  auto* latentT=inTensor(fg,"latent"); auto* timeT=inTensor(fg,"timestep");
  auto* capT=inTensor(fg,"cap_feats"); auto* capMaskT=inTensor(fg,"cap_mask");
  if(!latentT||!timeT||!capT||!capMaskT){ releaseSegment(*r,front); if(segCtx)S.api.contextFree(segCtx,nullptr); return e->NewStringUTF("frontend tensor layout missing"); }
  const bool fp32Graph = latentT->v1.dataType==QNN_DATATYPE_FLOAT_32;
  logline(*r,std::string("transformer frontend dtype=")+(fp32Graph?"fp32":"fp16"));

  std::vector<uint8_t> latentQ; packLatentBuffer(latentF,latentT->v1.dataType,latentQ);
  std::vector<uint8_t> timestepQ; packFloatBuffer(&jTimestep,1,timeT->v1.dataType,timestepQ);
  std::vector<uint8_t> capQ; packFloatBuffer(capF,512*2560,capT->v1.dataType,capQ);
  std::vector<float> maskInF(512); for(int i=0;i<512;i++) maskInF[i]=maskB[i]?1.0f:0.0f;
  std::vector<uint8_t> maskQ; packFloatBuffer(maskInF.data(),512,capMaskT->v1.dataType,maskQ);
  e->ReleaseFloatArrayElements(jLatent,latentF,JNI_ABORT);
  e->ReleaseFloatArrayElements(jCapFeats,capF,JNI_ABORT);
  e->ReleaseBooleanArrayElements(jCapMask,maskB,JNI_ABORT);

  auto* unifiedT=outTensor(fg,"unified"); auto* maskOutT=outTensor(fg,"unified_mask"); auto* adalnT=outTensor(fg,"adaln_input");
  if(!unifiedT||!maskOutT||!adalnT){ releaseSegment(*r,front); if(segCtx)S.api.contextFree(segCtx,nullptr); return e->NewStringUTF("frontend output tensor missing"); }
  const auto frontUnifiedDt=unifiedT->v1.dataType;
  const auto frontMaskDt=maskOutT->v1.dataType;
  const auto frontAdalnDt=adalnT->v1.dataType;
  std::vector<uint8_t> unifiedOut(bytes(*unifiedT)), maskOut(bytes(*maskOutT)), adalnOut(bytes(*adalnT));
  int64_t ms=0;
  if(!runGraph(*r,S,fg,{
        {latentT,latentQ.data(),latentQ.size()},
        {timeT,timestepQ.data(),timestepQ.size()},
        {capT,capQ.data(),capQ.size()},
        {capMaskT,maskQ.data(),maskQ.size()}},
      {{unifiedT,unifiedOut.data(),unifiedOut.size()},{maskOutT,maskOut.data(),maskOut.size()},{adalnT,adalnOut.data(),adalnOut.size()}},&ms)){
    std::string emsg="frontend failed: "+r->error; releaseSegment(*r,front); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
  report<<"frontend ms="<<ms<<" unified="<<tensorStatsRaw(*unifiedT,unifiedOut)<<" mask="<<tensorStatsRaw(*maskOutT,maskOut)<<" adaln="<<tensorStatsRaw(*adalnT,adalnOut)<<"\n";
  writeProgress(*r,"transformer: frontend 完成",1,14);
  { FILE* f=fopen((r->root+"/probe_frontend_unified.raw").c_str(),"wb"); if(f){ fwrite(unifiedOut.data(),1,unifiedOut.size(),f); fclose(f); } }
  releaseSegment(*r,front);
  if(segCtx){ S.api.contextFree(segCtx,nullptr); segCtx=nullptr; }

  std::vector<uint8_t> unifiedNext; transposeUnified(unifiedOut,frontUnifiedDt,unifiedNext);
  { FILE* f=fopen((r->root+"/probe_frontend_unified_layer_input.raw").c_str(),"wb"); if(f){ fwrite(unifiedNext.data(),1,unifiedNext.size(),f); fclose(f); } }
  std::vector<uint8_t> freqsQ(freqsElems*4);
  if(!readFile(r->root+"/assets/unified_freqs_f32_nhwc.raw",freqsQ))
    if(!readFile(r->root+"/unified_freqs_f32_nhwc.raw",freqsQ)){
      logline(*r,"freqs asset missing"); return e->NewStringUTF("freqs asset missing (assets/unified_freqs_f32_nhwc.raw)"); }
  const char* groupsFp16[]={"layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29"};
  const char* groupsFp32[]={"layers_00_02","layers_03_05","layers_06_08","layers_09_11","layers_12_14","layers_15_17","layers_18_20","layers_21_23","layers_24_26","layers_27_29"};
  const char** groups = fp32Graph ? groupsFp32 : groupsFp16;
  int groupCount = fp32Graph ? 10 : 5;
  int totalSegs = groupCount + 2;
  int segNo=2;
  for(int gi=0;gi<groupCount;gi++){
    const char* gname=groups[gi];
    SegmentGraph sg; sg.name=gname; sg.libName=std::string("libqnn_transformer_")+gname+".so";
    writeProgress(*r,std::string("transformer: 段 ")+std::to_string(segNo)+"/"+std::to_string(totalSegs)+" "+gname+" 编译中",segNo,totalSegs);
    segCtx=nullptr;
    if(S.api.contextCreate(S.backend,S.device,nullptr,&segCtx)!=QNN_SUCCESS){ logline(*r,std::string(gname)+" contextCreate failed"); return e->NewStringUTF((std::string(gname)+" contextCreate failed").c_str()); }
    if(!composeSegment(*r,sg,S,segCtx)){ std::string emsg=std::string(gname)+" compose failed: "+sg.error; releaseSegment(*r,sg); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
    auto& g=*sg.graphs[0];
    auto* uInT=inTensor(g,"unified_in"); auto* fInT=inTensor(g,"unified_freqs"); auto* mInT=inTensor(g,"unified_mask"); auto* aInT=inTensor(g,"adaln_input");
    auto* uOutT=outTensor(g,"unified_out");
    if(!uInT||!fInT||!mInT||!aInT||!uOutT){ releaseSegment(*r,sg); if(segCtx)S.api.contextFree(segCtx,nullptr); return e->NewStringUTF((std::string(gname)+" tensor missing").c_str()); }
    std::vector<uint8_t> maskNext; convertBuffer(maskOut,frontMaskDt,maskNext,mInT->v1.dataType);
    std::vector<uint8_t> adalnNext; convertBuffer(adalnOut,frontAdalnDt,adalnNext,aInT->v1.dataType);
    std::vector<uint8_t> layerOut(bytes(*uOutT));
    if(!runGraph(*r,S,g,{
          {uInT,unifiedNext.data(),unifiedNext.size()},
          {fInT,freqsQ.data(),freqsQ.size()},
          {mInT,maskNext.data(),maskNext.size()},
          {aInT,adalnNext.data(),adalnNext.size()}},
        {{uOutT,layerOut.data(),layerOut.size()}},&ms)){
      std::string emsg=std::string(gname)+" failed: "+r->error; releaseSegment(*r,sg); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
    report<<gname<<" ms="<<ms<<" out="<<tensorStatsRaw(*uOutT,layerOut)<<"\n";
    writeProgress(*r,std::string("transformer: 段 ")+std::to_string(segNo)+"/"+std::to_string(totalSegs)+" "+gname+" 完成",segNo,totalSegs);
    unifiedNext.swap(layerOut);
    releaseSegment(*r,sg);
    if(segCtx){ S.api.contextFree(segCtx,nullptr); segCtx=nullptr; }
    ++segNo;
  }
  SegmentGraph fseg; fseg.name="final"; fseg.libName="libqnn_transformer_final.so";
  writeProgress(*r,std::string("transformer: 段 ")+std::to_string(segNo)+"/"+std::to_string(totalSegs)+" final 编译中",segNo,totalSegs);
  segCtx=nullptr;
  if(S.api.contextCreate(S.backend,S.device,nullptr,&segCtx)!=QNN_SUCCESS){ logline(*r,"final contextCreate failed"); return e->NewStringUTF("final contextCreate failed"); }
  if(!composeSegment(*r,fseg,S,segCtx)){ std::string emsg="final compose failed: "+fseg.error; releaseSegment(*r,fseg); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
  auto& fgg=*fseg.graphs[0];
  auto* fuInT=inTensor(fgg,"unified_in"); auto* faInT=inTensor(fgg,"adaln_input"); auto* noiseT=outTensor(fgg,"noise_pred");
  if(!fuInT||!faInT||!noiseT){ releaseSegment(*r,fseg); if(segCtx)S.api.contextFree(segCtx,nullptr); return e->NewStringUTF("final tensor missing"); }
  std::vector<uint8_t> adalnFinal; convertBuffer(adalnOut,frontAdalnDt,adalnFinal,faInT->v1.dataType);
  std::vector<uint8_t> noiseQ(bytes(*noiseT));
  if(!runGraph(*r,S,fgg,{
        {fuInT,unifiedNext.data(),unifiedNext.size()},
        {faInT,adalnFinal.data(),adalnFinal.size()}},
      {{noiseT,noiseQ.data(),noiseQ.size()}},&ms)){
    std::string emsg="final failed: "+r->error; releaseSegment(*r,fseg); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
  report<<"final ms="<<ms<<" noise="<<tensorStatsRaw(*noiseT,noiseQ);
  writeProgress(*r,"transformer: 全部完成",segNo,totalSegs);
  FILE* fout=fopen((r->root+"/probe_noise_pred.raw").c_str(),"wb");
  if(fout){ fwrite(noiseQ.data(),1,noiseQ.size(),fout); fclose(fout); logline(*r,"dumped noise_pred to probe_noise_pred.raw"); }
  else logline(*r,"dump noise_pred failed");
  releaseSegment(*r,fseg);
  if(segCtx)S.api.contextFree(segCtx,nullptr);
  logline(*r,"transformer probe: "+report.str());
  return e->NewStringUTF(report.str().c_str());
}

// ---------------------------------------------------------------------------
// Qwen3 text encoder probe (uses fixed HTP backend r.hp)
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jstring JNICALL Java_com_example_zimage_ZImageRuntime_nativeTextEncoderProbe
  (JNIEnv* e, jobject, jlong h, jintArray jInputIds, jbooleanArray jAttention){
  auto* r=rt(h);
  QnnSet& S=r->hp;
  if(!(S.backend&&S.context)){ logline(*r,"text encoder unavailable: HTP backend not initialized"); return e->NewStringUTF("text encoder unavailable: HTP backend not initialized"); }
  jsize nIds=e->GetArrayLength(jInputIds), nAtt=e->GetArrayLength(jAttention);
  if(nIds!=512||nAtt!=512) return e->NewStringUTF("text encoder probe input size mismatch");
  jint* ids=e->GetIntArrayElements(jInputIds,nullptr);
  jboolean* att=e->GetBooleanArrayElements(jAttention,nullptr);
  if(!ids||!att){ if(ids)e->ReleaseIntArrayElements(jInputIds,ids,JNI_ABORT); if(att)e->ReleaseBooleanArrayElements(jAttention,att,JNI_ABORT); return e->NewStringUTF("text encoder array access failed"); }

  // embedding input: int64 [1,512]
  std::vector<uint8_t> idsQ(512*8);
  auto* id64=reinterpret_cast<int64_t*>(idsQ.data());
  for(int i=0;i<512;i++) id64[i]=(int64_t)ids[i];

  // attention_mask in QNN layout [1,512,512,1]: causal + padding.
  std::vector<uint8_t> maskQ(512*512*4);
  {
    auto* m=reinterpret_cast<float*>(maskQ.data());
    const float negInf=-3.402823466e38f;
    for(int q=0;q<512;q++) for(int k=0;k<512;k++)
      m[q*512+k]=(k>q||!att[k])?negInf:0.0f;
  }
  e->ReleaseIntArrayElements(jInputIds,ids,JNI_ABORT);
  e->ReleaseBooleanArrayElements(jAttention,att,JNI_ABORT);

  std::vector<uint8_t> cosQ(1*128*512*4), sinQ(1*128*512*4);
  if(!readFile(r->root+"/assets/cos_qnn_f32.raw",cosQ)||!readFile(r->root+"/assets/sin_qnn_f32.raw",sinQ)){
    logline(*r,"cos/sin assets missing"); return e->NewStringUTF("cos/sin assets missing under assets/"); }

  std::ostringstream report;
  int64_t ms=0;
  SegmentGraph emb; emb.name="embedding"; emb.libName="libqnn_embedding.so";
  writeProgress(*r,"text: 段 1/7 embedding 编译中",8,14);
  Qnn_ContextHandle_t segCtx=nullptr;
  if(S.api.contextCreate(S.backend,S.device,nullptr,&segCtx)!=QNN_SUCCESS){ logline(*r,"embedding contextCreate failed"); return e->NewStringUTF("embedding contextCreate failed"); }
  if(!composeSegment(*r,emb,S,segCtx)){ releaseSegment(*r,emb); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,"embedding failed: "+emb.error); return e->NewStringUTF(("embedding failed: "+emb.error).c_str()); }
  auto& eg=*emb.graphs[0];
  std::vector<uint8_t> embedOut(1*512*2560*4);
  if(!runGraph(*r,S,eg,{{inTensor(eg,"input_ids"),idsQ.data(),idsQ.size()}},
               {{outTensor(eg,"hidden_states"),embedOut.data(),embedOut.size()}},&ms)){
    std::string emsg="embedding failed: "+r->error; releaseSegment(*r,emb); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
  report<<"embedding ms="<<ms<<" out="<<f32Stats(embedOut)<<"\n";
  writeProgress(*r,"text: 段 1/7 embedding 完成",8,14);
  releaseSegment(*r,emb);
  if(segCtx)S.api.contextFree(segCtx,nullptr);

  std::vector<uint8_t> hiddenQ(1*2560*512*4);
  {
    const auto* src=reinterpret_cast<const float*>(embedOut.data());
    auto* dst=reinterpret_cast<float*>(hiddenQ.data());
    for(int s=0;s<512;s++) for(int d=0;d<2560;d++) dst[d*512+s]=src[s*2560+d];
  }

  const char* groups[]={"layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29","layers_30_35"};
  int textNo=2;
  std::vector<uint8_t> layerOut(1*2560*512*4);
  for(const char* gname:groups){
    SegmentGraph sg; sg.name=std::string("text_")+gname; sg.libName=std::string("libqnn_")+gname+".so";
    writeProgress(*r,std::string("text: 段 ")+std::to_string(textNo)+"/7 "+gname+" 编译中",9+textNo-2,14);
    segCtx=nullptr;
    if(S.api.contextCreate(S.backend,S.device,nullptr,&segCtx)!=QNN_SUCCESS){ logline(*r,std::string(gname)+" contextCreate failed"); return e->NewStringUTF((std::string(gname)+" contextCreate failed").c_str()); }
    if(!composeSegment(*r,sg,S,segCtx)){ std::string emsg=std::string(gname)+" compose failed: "+sg.error; releaseSegment(*r,sg); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
    auto& g=*sg.graphs[0];
    if(!runGraph(*r,S,g,{
          {inTensor(g,"hidden_states"),hiddenQ.data(),hiddenQ.size()},
          {inTensor(g,"attention_mask"),maskQ.data(),maskQ.size()},
          {inTensor(g,"cos"),cosQ.data(),cosQ.size()},
          {inTensor(g,"sin"),sinQ.data(),sinQ.size()}},
        {{outTensor(g,"hidden_states_out"),layerOut.data(),layerOut.size()}},&ms)){
      std::string emsg=std::string(gname)+" failed: "+r->error; releaseSegment(*r,sg); if(segCtx)S.api.contextFree(segCtx,nullptr); logline(*r,emsg); return e->NewStringUTF(emsg.c_str()); }
    report<<std::string("text_")+gname<<" ms="<<ms<<" out="<<f32Stats(layerOut)<<"\n";
    writeProgress(*r,std::string("text: 段 ")+std::to_string(textNo)+"/7 "+gname+" 完成",9+textNo-2,14);
    hiddenQ.swap(layerOut);
    releaseSegment(*r,sg);
    if(segCtx){ S.api.contextFree(segCtx,nullptr); segCtx=nullptr; }
    ++textNo;
  }
  report<<"final_hidden[-2]="<<f32Stats(hiddenQ);
  writeProgress(*r,"全部 probe 完成",14,14);
  FILE* fout=fopen((r->root+"/probe_hidden_fp32.raw").c_str(),"wb");
  if(fout){ fwrite(hiddenQ.data(),1,hiddenQ.size(),fout); fclose(fout); logline(*r,"dumped hidden to probe_hidden_fp32.raw"); }
  else logline(*r,"dump hidden failed");
  logline(*r,"text encoder probe: "+report.str());
  return e->NewStringUTF(report.str().c_str());
}

// ---------------------------------------------------------------------------
// Full pipeline: BPE tokenizer -> Qwen3 (HTP) -> DiT x8 steps (GPU) -> VAE (HTP)
// ---------------------------------------------------------------------------
namespace zpipe {

// ---- Byte-level GPT2/Qwen BPE --------------------------------------------
static const uint32_t ID_EOT=151643, ID_IM_START=151644, ID_IM_END=151645;

struct BpeModel{
  std::unordered_map<std::string,int> rankOf;   // merge rank by "a b"
  std::unordered_map<std::string,uint32_t> tokenId;
  bool load(const std::string& dir){
    { FILE* f=fopen((dir+"/qwen_vocab.tsv").c_str(),"rb"); if(!f) return false;
      char line[512];
      while(fgets(line,sizeof(line),f)){
        char* tab=strchr(line,'\t'); if(!tab) continue;
        *tab=0; char* tok=tab+1; size_t L=strlen(tok);
        while(L&&(tok[L-1]=='\n'||tok[L-1]=='\r')) tok[--L]=0;
        if(L) tokenId.emplace(tok,(uint32_t)atoi(line));
      }
      fclose(f); }
    { FILE* f=fopen((dir+"/qwen_merges.txt").c_str(),"rb"); if(!f) return false;
      char line[512]; int rank=0;
      while(fgets(line,sizeof(line),f)){
        char* sp=strchr(line,' '); if(!sp) continue;
        *sp=0; char* b=sp+1; size_t L=strlen(b);
        while(L&&(b[L-1]=='\n'||b[L-1]=='\r')) b[--L]=0;
        if(*line&&L) rankOf.emplace(std::string(line)+" "+std::string(b),rank++);
      }
      fclose(f); }
    return !tokenId.empty()&&!rankOf.empty();
  }
  std::vector<uint32_t> bpe(const std::string& piece) const{
    // Whole-piece fast path: most pretokenized pieces are single vocab entries.
    { auto it=tokenId.find(piece); if(it!=tokenId.end()) return {it->second}; }
    // Start from UTF-8 grapheme symbols (Ċ, Ġ, 'a', …), never raw bytes: the
    // vocab/merges operate on byte-unicode characters.
    std::vector<std::string> parts;
    for(size_t i=0;i<piece.size();){
      unsigned char c=(unsigned char)piece[i]; size_t n=1;
      if(c>=0xF0)n=4; else if(c>=0xE0)n=3; else if(c>=0xC0)n=2;
      if(i+n>piece.size()) n=1;
      parts.emplace_back(piece.substr(i,n)); i+=n;
    }
    while(parts.size()>1){
      int best=INT_MAX; size_t bi=SIZE_MAX;
      for(size_t i=0;i+1<parts.size();i++){
        auto it=rankOf.find(parts[i]+" "+parts[i+1]);
        if(it!=rankOf.end()&&it->second<best){ best=it->second; bi=i; }
      }
      if(bi==SIZE_MAX) break;
      parts[bi]=parts[bi]+parts[bi+1];
      parts.erase(parts.begin()+bi+1);
    }
    std::vector<uint32_t> out; out.reserve(parts.size());
    for(auto&p:parts){
      auto it=tokenId.find(p);
      out.push_back(it!=tokenId.end()?it->second:0);
    }
    return out;
  }
};

// Qwen pretokenizer regex equivalent (simplified but faithful for common
// prompts): contractions, letters, CJK chars, other letters, numbers <3 digits,
// spaces+letters, bytes. We implement the space-prefixed word + CJK + digit +
// single-byte fallback which covers natural-language prompts.
static void utf8Split(const std::string&s,std::vector<std::string>&out){
  for(size_t i=0;i<s.size();){
    unsigned char c=(unsigned char)s[i];
    size_t n=1;
    if(c>=0xF0)n=4; else if(c>=0xE0)n=3; else if(c>=0xC0)n=2;
    out.emplace_back(s.substr(i,n)); i+=n;
  }
}
static std::vector<std::string> pretokenize(const std::string& text){
  // GPT2-style regex equivalent: " ?letters+| ?digits{1,3}|other|spaces"
  // (space attaches to the FOLLOWING word). ASCII-focused; CJK chars fall
  // into the single-symbol branch.
  std::vector<std::string> pieces;
  size_t i=0; const size_t L=text.size();
  auto isSp=[&](size_t k){ return k<L&&(text[k]==' '||text[k]=='\t'); };
  auto isAlpha=[&](size_t k){ unsigned c=(unsigned char)text[k]; if(c<0x80) return isalpha(c)!=0; return true; };
  auto isDigit=[&](size_t k){ return k<L&&isdigit((unsigned char)text[k]); };
  while(i<L){
    if(isSp(i)){
      size_t j=i; while(isSp(j)) j++;
      if(j<L&&isAlpha(j)){ size_t k=j; while(k<L&&isAlpha(k)) k++; pieces.emplace_back(text.substr(i,k-i)); i=k; }
      else if(j<L&&isDigit(j)){ size_t k=j; while(k<L&&isDigit(k)) k++;
        for(size_t m=j;m<k;m+=3){ size_t e=std::min(m+3,k); pieces.emplace_back(text.substr(m,e-m)); }
        i=k; }
      else { pieces.emplace_back(text.substr(i,j+1-i)); i=j+1; }
    } else if(isAlpha(i)){
      size_t k=i; while(k<L&&isAlpha(k)) k++;
      pieces.emplace_back(text.substr(i,k-i)); i=k;
    } else if(isDigit(i)){
      size_t k=i; while(k<L&&isDigit(k)) k++;
      for(size_t m=i;m<k;m+=3){ size_t e2=std::min(m+3,k); pieces.emplace_back(text.substr(m,e2-m)); }
      i=k;
    } else {
      unsigned char c=(unsigned char)text[i]; size_t n=1;
      if(c>=0xF0)n=4; else if(c>=0xE0)n=3; else if(c>=0xC0)n=2;
      if(i+n>L) n=1;
      pieces.emplace_back(text.substr(i,n)); i+=n;
    }
  }
  return pieces;
}
// byte-level mapping used by GPT2/Qwen: printable unicode chars in vocab are
// the 256 bytes remapped. The vocab file stores them already as unicode chars
// (Ġ=0x20 etc). For ASCII range they are identity except control chars. We use
// identity for printable ASCII and map other bytes via latin1→U+0100 style.
static void toByteUnicode(const std::string& raw,std::string& out){
  static const char* hexlo="?";(void)hexlo;
  for(unsigned char c:raw){
    if(c==' '){ out+="Ġ"; continue; }
    if(c=='\n'){ out+="Ċ"; continue; }
    if(c=='\t'){ out+="ĉ"; continue; }
    if(c=='\r'){ out+="Č"; continue; }
    if(c>=33&&c<=126&&c!='!'||true){ char buf[8]; int L=0;
      // identity for printable ASCII
      if(c>=32&&c<127&&!(c==0x20)){ buf[0]=(char)c;buf[1]=0; out+=buf; continue; }
      // non-printable / high bytes: map to U+0100+c (vocab uses Ā..ÿ range)
      unsigned cp=0x100+c;
      if(cp<0x800){ buf[L++]=(char)(0xC0|(cp>>6)); buf[L++]=(char)(0x80|(cp&0x3F)); }
      else { buf[L++]=(char)(0xE0|(cp>>12)); buf[L++]=(char)(0x80|((cp>>6)&0x3F)); buf[L++]=(char)(0x80|(cp&0x3F)); }
      buf[L]=0; out+=buf;
    }
  }
}
static bool encodePrompt(BpeModel& bm,const std::string& prompt,
                         std::vector<int>& ids,std::vector<uint8_t>& mask){
  ids.clear(); mask.assign(512,0);
  auto push=[&](uint32_t id){ ids.push_back((int)id); };
  auto pushText=[&](const std::string& s){
    for(auto&pc:bm.bpe(s)) push(pc);
  };
  // Template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
  std::string nl; toByteUnicode("\n",nl); // "Ċ"
  push(ID_IM_START); pushText("user"); pushText(nl);
  for(auto&piece:pretokenize(prompt)){
    std::string enc; toByteUnicode(piece,enc);
    pushText(enc);
  }
  push(ID_IM_END); pushText(nl);
  push(ID_IM_START); pushText("assistant"); pushText(nl);
  while((int)ids.size()<512) push(ID_EOT);
  ids.resize(512);
  for(int i=0;i<512;i++) mask[i]=(ids[i]!=ID_EOT)?1:0;
  return true;
}

// ---- FlowMatch Euler scheduler (shift=3, 8 steps) -------------------------
static void buildSigmas(int steps,double shift,std::vector<double>& sigmas){
  sigmas.resize(steps+1);
  for(int i=0;i<steps;i++){
    double s=1.0-(double)i/steps;             // linspace(1,1/N,N)
    sigmas[i]=shift*s/(1.0+(shift-1.0)*s);    // shift transform
  }
  sigmas[steps]=0.0;
}

// ---- RNG ------------------------------------------------------------------
static void fillRandn(std::vector<float>& v,uint64_t seed){
  std::mt19937_64 g(seed?seed:(uint64_t)std::random_device{}());
  std::normal_distribution<float> d(0.0f,1.0f);
  for(auto&x:v) x=d(g);
}

} // namespace zpipe

// Build (once) the transformer segment cache on r.main: frontend + layer
// groups + final, each with its own QNN context held open for reuse across
// the denoise steps. Returns false on first-call failure.
//
// Memory strategy (LMK workaround): after compiling a segment we export a QNN
// context binary to disk and FREE the compiled context. On subsequent runs
// segments are restored with contextCreateFromBinary. The full 12-segment
// resident set was measured at RSS 8.5GB and killed by ColorOS LMK; binaries
// on disk + mmap'd weights keep the anonymous footprint far lower.
static bool segBinPath(Runtime&r,const std::string& name,std::string& out){
  std::string priv=r.root+"/segbin/"+name+".ctxbin";
  FILE* f=fopen(priv.c_str(),"rb");
  if(f){ fclose(f); out=priv; return true; }
  // fallback: precompiled binaries staged in app-external dir (generated by
  // qnn-context-binary-generator on host; avoids in-app compile memory spikes)
  out="/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/segbin/"+name+".ctxbin";
  f=fopen(out.c_str(),"rb");
  if(f){ fclose(f); return true; }
  out=priv;
  return false;
}
// Ensure the private segbin dir exists before cacheSaveBinaryImpl writes there.
static bool ensureSegbinDir(Runtime&r){
  std::string d=r.root+"/segbin";
  return mkdir(d.c_str(),0700)==0||errno==EEXIST;
}
// Generic binary save/load helpers shared by transformer and text caches.
struct MappedFile{ int fd=-1; void* addr=nullptr; size_t size=0;
  bool map(const std::string& path){
    fd=open(path.c_str(),O_RDONLY);
    if(fd<0) return false;
    off_t len=lseek(fd,0,SEEK_END); lseek(fd,0,SEEK_SET);
    if(len<=0){ close(fd); fd=-1; return false; }
    size=(size_t)len;
    addr=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
    if(addr==MAP_FAILED){ addr=nullptr; close(fd); fd=-1; size=0; return false; }
    return true;
  }
  void unmap(){
    if(addr&&size) munmap(addr,size);
    if(fd>=0) close(fd);
    addr=nullptr; size=0; fd=-1;
  }
  ~MappedFile(){ unmap(); }
};
static bool cacheLoadSegment(Runtime&r,QnnSet&S,const std::string& libPrefix,const std::string& name,
                             SegmentGraph& sg,Qnn_ContextHandle_t& ctxOut,std::string& errOut){
  std::string bin; segBinPath(r,name,bin);
  // mmap and KEEP the mapping alive until the segment is released: QNN may
  // lazily reference the buffer after createFromBinary returns (premature
  // munmap caused SIGSEGV in the field).
  sg.unmapBin();
  sg.binFd=open(bin.c_str(),O_RDONLY);
  if(sg.binFd<0){ errOut="no binary: "+bin; return false; }
  off_t len=lseek(sg.binFd,0,SEEK_END); lseek(sg.binFd,0,SEEK_SET);
  if(len<=0){ close(sg.binFd); sg.binFd=-1; errOut="empty binary: "+bin; return false; }
  sg.binSize=(size_t)len;
  sg.binAddr=mmap(nullptr,sg.binSize,PROT_READ,MAP_PRIVATE,sg.binFd,0);
  if(sg.binAddr==MAP_FAILED){ sg.binAddr=nullptr; close(sg.binFd); sg.binFd=-1; sg.binSize=0; errOut="mmap failed: "+bin; return false; }
  if(S.api.contextCreateFromBinary(S.backend,S.device,nullptr,sg.binAddr,(Qnn_ContextBinarySize_t)sg.binSize,&ctxOut,nullptr)!=QNN_SUCCESS){
    sg.unmapBin(); errOut=name+" createFromBinary failed"; return false;
  }
  sg.name=name;
  auto mkpath=[&](const std::string& lib){ return r.root+"/lib/arm64-v8a/"+lib; };
  std::string libA=libPrefix+name+".so";
  std::string base=name.substr(0,5)=="text_"?name.substr(5):name;
  std::string libB=libPrefix+base+".so";
  { FILE* f=fopen(mkpath(libB).c_str(),"rb"); if(f){ fclose(f); libA=libB; } }
  sg.libName=libA;
  sg.lib=load(mkpath(sg.libName));
  if(!sg.lib){ errOut="dlopen failed: "+mkpath(sg.libName); return false; }
  // Binary-restored segment: graph handle comes from the QNN context via
  // graphRetrieve. freeGraphs must stay null (releaseSegment deletes our shell).
  sg.compose=nullptr;
  sg.freeGraphs=nullptr;
  GraphInfo* ginfo=new GraphInfo();
  memset(ginfo,0,sizeof(GraphInfo));
  ginfo->graphName=const_cast<char*>("model");
  if(S.api.graphRetrieve(ctxOut,ginfo->graphName,&ginfo->graph)!=QNN_SUCCESS||!ginfo->graph){
    errOut=name+" graphRetrieve(binary) failed"; return false;
  }
  sg.graphs=new GraphInfo*(ginfo);
  sg.count=1;
  logline(r,"rss after restore "+name+": "+std::to_string(readSelfRssMB())+"MB ["+readSmaps()+"]");
  // Fill tensor metadata via libQnnSystem so inTensor/outTensor work.
  using ProvidersSysFn=Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t***,uint32_t*);
  void* sys=qnnSystemLib();
  if(sys){
    auto gp=reinterpret_cast<ProvidersSysFn>(dlsym(sys,"QnnSystemInterface_getProviders"));
    const QnnSystemInterface_t** provs=nullptr; uint32_t nprov=0;
    QnnSystemContext_Handle_t h=nullptr;
    const QnnSystemContext_BinaryInfo_t* info=nullptr; Qnn_ContextBinarySize_t infoSize=0;
    Qnn_ErrorHandle_t grc=gp?gp(&provs,&nprov):1;
    bool fns = provs&&nprov>0&&provs[0]->v1_13.systemContextCreate&&provs[0]->v1_13.systemContextGetBinaryInfo;
    bool haveIfc = gp && grc==QNN_SUCCESS && fns;
    // reuse sg.binAddr mapping (still alive); do NOT open a second mapping here.
    Qnn_ErrorHandle_t crc= haveIfc?provs[0]->v1_13.systemContextCreate(&h):1;
    Qnn_ErrorHandle_t grc2= (crc==QNN_SUCCESS&&h&&sg.binAddr)?provs[0]->v1_13.systemContextGetBinaryInfo(h,sg.binAddr,sg.binSize,&info,&infoSize):1;
    if(haveIfc && sg.binAddr && crc==QNN_SUCCESS && h && grc2==QNN_SUCCESS && info
       && (info->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1||info->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)){
      uint32_t ng=info->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1
                    ?info->contextBinaryInfoV1.numGraphs:info->contextBinaryInfoV3.numGraphs;
      auto graphs =info->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1
                    ?info->contextBinaryInfoV1.graphs :info->contextBinaryInfoV3.graphs;
      if(ng>0&&graphs){
        auto& gv1=graphs[0].graphInfoV1;
        ginfo->numInputTensors=gv1.numGraphInputs;
        ginfo->inputTensors=gv1.graphInputs;   // owned by system ctx; intentionally never freed
        ginfo->numOutputTensors=gv1.numGraphOutputs;
        ginfo->outputTensors=gv1.graphOutputs;
        logline(r,std::string("binaryInfo ")+name+": in="+std::to_string(gv1.numGraphInputs)+" out="+std::to_string(gv1.numGraphOutputs));
      }
    } else {
      logline(r,std::string("binaryInfo unavailable for ")+name);
    }
  }
  return true;
}
static bool cacheSaveBinaryImpl(QnnSet&S,const std::string& bin,Qnn_ContextHandle_t ctx){
  { // make sure parent dir exists (fopen fails silently otherwise)
    auto slash=bin.find_last_of('/');
    if(slash!=std::string::npos){ std::string d=bin.substr(0,slash); mkdir(d.c_str(),0700); }
  }
  Qnn_ContextBinarySize_t size=0;
  if(S.api.contextGetBinarySize(ctx,&size)!=QNN_SUCCESS||!size) return false;
  std::vector<uint8_t> buf((size_t)size);
  Qnn_ContextBinarySize_t written=0;
  if(S.api.contextGetBinary(ctx,buf.data(),buf.size(),&written)!=QNN_SUCCESS) return false;
  FILE* f=fopen(bin.c_str(),"wb"); if(!f) return false;
  fwrite(buf.data(),1,(size_t)written,f); fclose(f);
  return true;
}
static bool saveSegmentBinary(Runtime&r,QnnSet&S,const std::string& name,Qnn_ContextHandle_t ctx){
  std::string bin; segBinPath(r,name,bin);
  Qnn_ContextBinarySize_t size=0;
  if(S.api.contextGetBinarySize(ctx,&size)!=QNN_SUCCESS||!size) return false;
  std::vector<uint8_t> buf((size_t)size);
  Qnn_ContextBinarySize_t written=0;
  if(S.api.contextGetBinary(ctx,buf.data(),buf.size(),&written)!=QNN_SUCCESS) return false;
  FILE* f=fopen(bin.c_str(),"wb"); if(!f) return false;
  fwrite(buf.data(),1,(size_t)written,f); fclose(f);
  logline(r,"saved "+name+" ctxbin "+std::to_string((long long)written/1048576)+"MB");
  return true;
}
static bool ensureTransformerCache(Runtime&r,std::string& errOut){
  if(r.tcacheReady) return true;
  if(!r.tcacheError.empty()){ errOut=r.tcacheError; return false; }
  QnnSet&S=r.main;
  const int N=12; // frontend + 10 layers + final (fp32); fp16 uses fewer but slots are fine
  r.tcache.resize(N);
  auto mk=[&](int idx,const char* name,const char* lib)->bool{
    r.tcache[idx].name=name; r.tcache[idx].libName=lib;
    std::string bin; segBinPath(r,name,bin);
    FILE* probe=fopen(bin.c_str(),"rb");
    bool haveBin=(probe!=nullptr); if(probe)fclose(probe);
    if(haveBin){
      // File-presence check ONLY. Do NOT createFromBinary here: restoring a
      // 2.2GB GPU context per segment (even briefly) spikes RSS and LMK-kills
      // the app when done 12x in a row. Actual restore happens per-segment in
      // pipeTransformer (step-restore). fp32 dtype known from build config.
      off_t sz=0; { FILE* f=fopen(bin.c_str(),"rb"); if(f){ fseek(f,0,SEEK_END); sz=ftell(f); fclose(f);} }
      if(sz>1000000){
        r.tcCtx.push_back(nullptr);
        r.tcacheStepRestore=true;
        logline(r,std::string("tcache file-ok ")+name+" ("+std::to_string(sz/1048576)+"MB), step-restore");
        return true;
      }
      logline(r,std::string("tcache binary too small for ")+name+" (recompiling)");
    }
    Qnn_ContextHandle_t ctx=nullptr;
    if(S.api.contextCreate(S.backend,S.device,nullptr,&ctx)!=QNN_SUCCESS){ errOut=std::string(name)+" contextCreate failed"; return false; }
    if(!composeSegment(r,r.tcache[idx],S,ctx)){ errOut=std::string(name)+" compose: "+r.tcache[idx].error; S.api.contextFree(ctx,nullptr); return false; }
    saveSegmentBinary(r,S,name,ctx);
    // Keep the FIRST segment alive only while needed? We free all contexts to
    // stay under the LMK limit; pipeTransformer restores per-step instead.
    S.api.contextFree(ctx,nullptr); ctx=nullptr;
    releaseSegment(r,r.tcache[idx]);
    r.tcCtx.push_back(nullptr);
    r.tcacheStepRestore=true;
    logline(r,std::string("tcache compile+save ")+name+", context freed (step-restore mode)");
    return true;
  };
  writeProgress(r,"pipeline: 编译 transformer 段(一次性)",15,100);
  logline(r,"tcache: entering frontend step");
  if(!mk(0,"frontend","libqnn_transformer_frontend.so")) goto fail;
  logline(r,"tcache: frontend ok");
  {
    // NOTE: after mk() the frontend segment is already released (step-restore
    // verify path frees context + graphs). Use dtype captured inside mk().
    bool fp32Graph=r.tcacheFrontendFp32;
    const char* groupsFp16[]={"layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29"};
    const char* groupsFp32[]={"layers_00_02","layers_03_05","layers_06_08","layers_09_11","layers_12_14","layers_15_17","layers_18_20","layers_21_23","layers_24_26","layers_27_29"};
    const char** groups = fp32Graph ? groupsFp32 : groupsFp16;
    int groupCount = fp32Graph ? 10 : 5;
    for(int gi=0;gi<groupCount;gi++){
      std::string nm=groups[gi]; std::string lib="libqnn_transformer_"+nm+".so";
      writeProgress(r,"pipeline: 编译 "+nm+" ("+std::to_string(gi+1)+"/"+std::to_string(groupCount)+")",16+gi*40/groupCount,100);
      logline(r,"tcache prepare "+nm);
      if(!mk(1+gi,nm.c_str(),lib.c_str())) goto fail;
    }
    int finalIdx=1+groupCount;
    if(!mk(finalIdx,"final","libqnn_transformer_final.so")) goto fail;
    r.tcache.resize(finalIdx+1);
    r.tcacheGroupCount=groupCount;
    logline(r,"tcache ready: "+std::to_string(finalIdx+1)+" segments (stepRestore="+std::to_string(r.tcacheStepRestore)+")");
    r.tcacheReady=true;
    return true;
  }
fail:
  r.tcacheError=errOut;
  logline(r,"tcache failed: "+errOut);
  return false;
}

// Run one transformer forward using the persistent cache.
// REBUILD EXPERIMENT: after N segment frees, tear down and recreate the whole
// GPU backend (backendFree+deviceFree+dlclose). If the kgsl dma-buf cache is
// attached to the backend/device/lib lifetime, RSS drops; if it is process-
// global, RSS stays and the GPU route is dead.
static bool rebuildGpuBackend(Runtime& r){
  logline(r,"rebuild-gpu: tearing down backend (rss="+std::to_string(readSelfRssMB())+"MB ["+readSmaps()+"])");
  r.main.release(&r);
  logline(r,"rebuild-gpu: backend freed, rss="+std::to_string(readSelfRssMB())+"MB ["+readSmaps()+"]");
  bool ok=makeBackend(r,r.main,"libQnnGpu.so",false);
  logline(r,ok?("rebuild-gpu: backend recreated, rss="+std::to_string(readSelfRssMB())+"MB ["+readSmaps()+"]")
               :std::string("rebuild-gpu: RECREATE FAILED: ")+r.error);
  return ok;
}
static bool pipeTransformer(Runtime&r,const std::vector<float>& latentCHW,float tNorm,
                            const std::vector<float>& capF,const std::vector<uint8_t>& capMaskBool,
                            std::vector<float>& noiseCHW,std::string& errOut){
  QnnSet&S=r.main;
  if(!ensureTransformerCache(r,errOut)) return false;
  // Step-restore mode (LMK workaround): at most ONE segment context is alive
  // at a time. Restore -> execute -> free, per segment, per step.
  if(r.tcacheStepRestore){
    auto restore=[&](int idx,std::string& e2)->bool{
      Qnn_ContextHandle_t ctx=nullptr;
      if(cacheLoadSegment(r,S,std::string("libqnn_transformer_"),r.tcache[idx].name,r.tcache[idx],ctx,e2)) return true;
      return false;
    };
    int segFreed=0;
    auto freeSeg=[&](int idx){
      releaseSegment(r,r.tcache[idx]);
      if(r.tcCtx[idx]&&S.api.contextFree){ S.api.contextFree(r.tcCtx[idx],nullptr); r.tcCtx[idx]=nullptr; }
      segFreed++;
      if(segFreed>=4 && r.backendKind==2){
        segFreed=0;
        if(!rebuildGpuBackend(r)) logline(r,"rebuild-gpu: subsequent segments will fail!");
      }
    };
    if(!restore(0,errOut)) return false;
    auto& fg=*r.tcache[0].graphs[0];
    auto* latentT=inTensor(fg,"latent"); auto* timeT=inTensor(fg,"timestep");
    auto* capT=inTensor(fg,"cap_feats"); auto* capMaskT=inTensor(fg,"cap_mask");
    auto* unifiedT=outTensor(fg,"unified"); auto* maskOutT=outTensor(fg,"unified_mask"); auto* adalnT=outTensor(fg,"adaln_input");
    if(!latentT||!timeT||!capT||!capMaskT||!unifiedT||!maskOutT||!adalnT){ freeSeg(0); errOut="frontend tensor missing"; return false; }
    const auto dt=latentT->v1.dataType;
    std::vector<uint8_t> latentQ; packLatentBuffer(latentCHW.data(),dt,latentQ);
    std::vector<uint8_t> timeQ; packFloatBuffer(&tNorm,1,timeT->v1.dataType,timeQ);
    std::vector<uint8_t> capQ; packFloatBuffer(capF.data(),512*2560,capT->v1.dataType,capQ);
    std::vector<float> maskInF(512); for(int i=0;i<512;i++) maskInF[i]=capMaskBool[i]?1.0f:0.0f;
    std::vector<uint8_t> maskQ; packFloatBuffer(maskInF.data(),512,capMaskT->v1.dataType,maskQ);
    std::vector<uint8_t> unifiedOut(bytes(*unifiedT)), maskOut(bytes(*maskOutT)), adalnOut(bytes(*adalnT));
    int64_t ms=0;
    bool ok=runGraph(r,S,fg,{
          {latentT,latentQ.data(),latentQ.size()},
          {timeT,timeQ.data(),timeQ.size()},
          {capT,capQ.data(),capQ.size()},
          {capMaskT,maskQ.data(),maskQ.size()}},
        {{unifiedT,unifiedOut.data(),unifiedOut.size()},{maskOutT,maskOut.data(),maskOut.size()},{adalnT,adalnOut.data(),adalnOut.size()}},&ms);
    logline(r,"pipeline frontend ms="+std::to_string(ms));
    freeSeg(0);
    if(!ok){ errOut="frontend exec: "+r.error; return false; }
    std::vector<uint8_t> unifiedNext; transposeUnified(unifiedOut,dt,unifiedNext);
    static std::vector<uint8_t> freqsQ;
    if(freqsQ.empty()){
      if(!readFile(r.root+"/assets/unified_freqs_f32_nhwc.raw",freqsQ)&&!readFile(r.root+"/unified_freqs_f32_nhwc.raw",freqsQ)){ errOut="freqs asset missing"; return false; }
    }
    int groupCount=r.tcacheGroupCount;
    for(int gi=0;gi<groupCount;gi++){
      int idx=1+gi;
      std::string e2;
      if(!restore(idx,e2)){ errOut=r.tcache[idx].name+" restore: "+e2; return false; }
      auto& g=*r.tcache[idx].graphs[0];
      auto* uInT=inTensor(g,"unified_in"); auto* fInT=inTensor(g,"unified_freqs"); auto* mInT=inTensor(g,"unified_mask"); auto* aInT=inTensor(g,"adaln_input");
      auto* uOutT=outTensor(g,"unified_out");
      if(!uInT||!fInT||!mInT||!aInT||!uOutT){ freeSeg(idx); errOut=r.tcache[idx].name+" tensor missing"; return false; }
      std::vector<uint8_t> maskNext; convertBuffer(maskOut,maskOutT->v1.dataType,maskNext,mInT->v1.dataType);
      std::vector<uint8_t> adalnNext; convertBuffer(adalnOut,adalnT->v1.dataType,adalnNext,aInT->v1.dataType);
      std::vector<uint8_t> layerOut(bytes(*uOutT));
      ok=runGraph(r,S,g,{
            {uInT,unifiedNext.data(),unifiedNext.size()},
            {fInT,freqsQ.data(),freqsQ.size()},
            {mInT,maskNext.data(),maskNext.size()},
            {aInT,adalnNext.data(),adalnNext.size()}},
          {{uOutT,layerOut.data(),layerOut.size()}},&ms);
      unifiedNext.swap(layerOut);
      freeSeg(idx);
      if(!ok){ errOut=r.tcache[idx].name+" exec: "+r.error; return false; }
    }
    {
      int idx=(int)r.tcache.size()-1;
      std::string e2;
      if(!restore(idx,e2)){ errOut="final restore: "+e2; return false; }
      auto& fgg=*r.tcache[idx].graphs[0];
      auto* fuInT=inTensor(fgg,"unified_in"); auto* faInT=inTensor(fgg,"adaln_input"); auto* noiseT=outTensor(fgg,"noise_pred");
      if(!fuInT||!faInT||!noiseT){ freeSeg(idx); errOut="final tensor missing"; return false; }
      std::vector<uint8_t> adalnFinal; convertBuffer(adalnOut,dt,adalnFinal,faInT->v1.dataType);
      std::vector<uint8_t> noiseQ(bytes(*noiseT));
      ok=runGraph(r,S,fgg,{
            {fuInT,unifiedNext.data(),unifiedNext.size()},
            {faInT,adalnFinal.data(),adalnFinal.size()}},
          {{noiseT,noiseQ.data(),noiseQ.size()}},&ms);
      freeSeg(idx);
      if(!ok){ errOut="final exec: "+r.error; return false; }
      noiseCHW.resize(16*64*64);
      for(size_t i=0;i<noiseCHW.size();i++) noiseCHW[i]=elemToF32(noiseQ,noiseT->v1.dataType,i);
    }
    return true;
  }
  // Resident mode: all segment contexts stay alive across steps.
  auto& fg=*r.tcache[0].graphs[0];
  auto* latentT=inTensor(fg,"latent"); auto* timeT=inTensor(fg,"timestep");
  auto* capT=inTensor(fg,"cap_feats"); auto* capMaskT=inTensor(fg,"cap_mask");
  auto* unifiedT=outTensor(fg,"unified"); auto* maskOutT=outTensor(fg,"unified_mask"); auto* adalnT=outTensor(fg,"adaln_input");
  if(!latentT||!timeT||!capT||!capMaskT||!unifiedT||!maskOutT||!adalnT){ errOut="frontend tensor missing"; return false; }
  const auto dt=latentT->v1.dataType;
  std::vector<uint8_t> latentQ; packLatentBuffer(latentCHW.data(),dt,latentQ);
  std::vector<uint8_t> timeQ; packFloatBuffer(&tNorm,1,timeT->v1.dataType,timeQ);
  std::vector<uint8_t> capQ; packFloatBuffer(capF.data(),512*2560,capT->v1.dataType,capQ);
  std::vector<float> maskInF(512); for(int i=0;i<512;i++) maskInF[i]=capMaskBool[i]?1.0f:0.0f;
  std::vector<uint8_t> maskQ; packFloatBuffer(maskInF.data(),512,capMaskT->v1.dataType,maskQ);
  std::vector<uint8_t> unifiedOut(bytes(*unifiedT)), maskOut(bytes(*maskOutT)), adalnOut(bytes(*adalnT));
  int64_t ms=0;
  if(!runGraph(r,S,fg,{
        {latentT,latentQ.data(),latentQ.size()},
        {timeT,timeQ.data(),timeQ.size()},
        {capT,capQ.data(),capQ.size()},
        {capMaskT,maskQ.data(),maskQ.size()}},
      {{unifiedT,unifiedOut.data(),unifiedOut.size()},{maskOutT,maskOut.data(),maskOut.size()},{adalnT,adalnOut.data(),adalnOut.size()}},&ms)){
    errOut="frontend exec: "+r.error; return false; }
  logline(r,"pipeline frontend ms="+std::to_string(ms));

  std::vector<uint8_t> unifiedNext; transposeUnified(unifiedOut,unifiedT->v1.dataType,unifiedNext);
  static std::vector<uint8_t> freqsQ;
  if(freqsQ.empty()){
    if(!readFile(r.root+"/assets/unified_freqs_f32_nhwc.raw",freqsQ)&&!readFile(r.root+"/unified_freqs_f32_nhwc.raw",freqsQ)){ errOut="freqs asset missing"; return false; }
  }

  int groupCount=(int)r.tcache.size()-2; // minus frontend and final
  for(int gi=0;gi<groupCount;gi++){
    auto& sg=r.tcache[1+gi];
    auto& g=*sg.graphs[0];
    auto* uInT=inTensor(g,"unified_in"); auto* fInT=inTensor(g,"unified_freqs"); auto* mInT=inTensor(g,"unified_mask"); auto* aInT=inTensor(g,"adaln_input");
    auto* uOutT=outTensor(g,"unified_out");
    if(!uInT||!fInT||!mInT||!aInT||!uOutT){ errOut=sg.name+" tensor missing"; return false; }
    std::vector<uint8_t> maskNext; convertBuffer(maskOut,maskOutT->v1.dataType,maskNext,mInT->v1.dataType);
    std::vector<uint8_t> adalnNext; convertBuffer(adalnOut,adalnT->v1.dataType,adalnNext,aInT->v1.dataType);
    std::vector<uint8_t> layerOut(bytes(*uOutT));
    if(!runGraph(r,S,g,{
          {uInT,unifiedNext.data(),unifiedNext.size()},
          {fInT,freqsQ.data(),freqsQ.size()},
          {mInT,maskNext.data(),maskNext.size()},
          {aInT,adalnNext.data(),adalnNext.size()}},
        {{uOutT,layerOut.data(),layerOut.size()}},&ms)){
      errOut=sg.name+" exec: "+r.error; return false; }
    unifiedNext.swap(layerOut);
  }
  {
    auto& fseg=r.tcache.back();
    auto& fgg=*fseg.graphs[0];
    auto* fuInT=inTensor(fgg,"unified_in"); auto* faInT=inTensor(fgg,"adaln_input"); auto* noiseT=outTensor(fgg,"noise_pred");
    if(!fuInT||!faInT||!noiseT){ errOut="final tensor missing"; return false; }
    std::vector<uint8_t> adalnFinal; convertBuffer(adalnOut,adalnT->v1.dataType,adalnFinal,faInT->v1.dataType);
    std::vector<uint8_t> noiseQ(bytes(*noiseT));
    if(!runGraph(r,S,fgg,{
          {fuInT,unifiedNext.data(),unifiedNext.size()},
          {faInT,adalnFinal.data(),adalnFinal.size()}},
        {{noiseT,noiseQ.data(),noiseQ.size()}},&ms)){
      errOut="final exec: "+r.error; return false; }
    noiseCHW.resize(16*64*64);
    for(size_t i=0;i<noiseCHW.size();i++) noiseCHW[i]=elemToF32(noiseQ,noiseT->v1.dataType,i);
  }
  return true;
}

// ---------------------------------------------------------------------------
// JNI entry points
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jlong JNICALL Java_com_example_zimage_ZImageRuntime_nativeCreate(JNIEnv*e,jobject,jstring root,jstring backend){
  const char*c=e->GetStringUTFChars(root,nullptr);
  auto*r=new Runtime();
  r->root=c?c:"";e->ReleaseStringUTFChars(root,c);
  if(backend){const char*b=e->GetStringUTFChars(backend,nullptr);if(b){if(std::strcmp(b,"cpu")==0)r->backendKind=1;else if(std::strcmp(b,"gpu")==0)r->backendKind=2;}if(b)e->ReleaseStringUTFChars(backend,b);}
  {FILE*f=fopen((r->root+"/jni.log").c_str(),"w");if(f)fclose(f);}
  if(!g_sysLib) g_sysLib=load(r->root+"/lib/arm64-v8a/libQnnSystem.so");
  logline(*r,std::string("=== zimage runtime init === transformer backend=")+backendLabel(r->backendKind)+" (VAE/text fixed HTP)");
  if(r->backendKind==0){
    // Single-backend mode: everything on HTP in this thread.
    init(*r);
    if(r->hp.backend&&r->hp.context) composeVae(*r,r->hp);
    else logline(*r,"WARN: VAE not composed (HTP unavailable).");
  }else{
    // Ordered single-thread attempt: create AND compose the HTP set first,
    // while it is the ONLY backend in the process; only then bring up the
    // transformer backend. (Same-thread second-backend, swapped order,
    // RTLD_LOCAL and separate-thread variants all fail with
    // MODEL_GRAPH_ERROR=4 — adapter state looks process-global.)
    r->hp.label="HTP2";
    bool hpOk=makeBackend(*r,r->hp,"libQnnHtp.so",true);
    if(!hpOk){ r->error.clear(); }
    if(hpOk&&r->hp.backend&&r->hp.context){
      composeVae(*r,r->hp); // must run before the second backend exists
      logline(*r,std::string("VAE pre-compose on dedicated HTP: ")+(r->graphs?"OK":"FAIL "+r->error));
      r->error.clear();
    }
    if(r->backendKind==2){
      r->main.label="GPU"; openClProbe(*r);
      if(!makeBackend(*r,r->main,"libQnnGpu.so",false)) return reinterpret_cast<jlong>(r);
    }else{
      r->main.label="CPU";
      if(!makeBackend(*r,r->main,"libQnnCpu.so",false)) return reinterpret_cast<jlong>(r);
    }
    if(!(hpOk&&r->hp.backend)){
      r->hp=r->main; r->hp.alias=true;
      logline(*r,"NOTE: degraded to single-backend; VAE/text on "+r->hp.label);
    }else{
      logline(*r,"NOTE: dual-backend mode (ordered): VAE/text composed on dedicated HTP");
    }
    r->api=r->main.api; r->backend=r->main.backend; r->device=r->main.device; r->context=r->main.context; r->logHandle=r->main.log;
  }
  return reinterpret_cast<jlong>(r);
}
extern "C" JNIEXPORT jstring JNICALL Java_com_example_zimage_ZImageRuntime_nativeStatus(JNIEnv*e,jobject,jlong h){auto*r=rt(h);std::ostringstream o;o<<"Runtime root: "<<r->root<<"\nSystem: "<<(r->sys?"loaded":"missing")<<"\nGPU lib: "<<(r->gpu?"loaded":"missing")<<" main="<<r->main.label<<" ("<<(r->main.backend?"ok":"-")<<")\nHTP lib: "<<(r->htp?"loaded":"missing")<<(r->hp.backend?" (hp ok)":" (hp missing)")<<"\nVAE library: "<<r->modelName<<" ("<<(r->model?"loaded":"missing")<<")"<<"\nVAE graph: "<<(r->graphs?r->info:(r->error.empty()?"not composed":r->error));logline(*r,"status: "+o.str());return e->NewStringUTF(o.str().c_str());}
std::string tensorFloatStats(const Qnn_Tensor_t& t, const std::vector<uint8_t>& buf){
  size_t n=0; float mn=0.0f, mx=0.0f; double sum=0.0;
  if(t.v1.dataType==QNN_DATATYPE_FLOAT_32){ const auto* v=reinterpret_cast<const float*>(buf.data()); n=buf.size()/4; if(!n)return "empty"; mn=mx=v[0]; for(size_t i=0;i<n;i++){float x=v[i];mn=std::min(mn,x);mx=std::max(mx,x);sum+=x;} }
  else if(t.v1.dataType==QNN_DATATYPE_FLOAT_16){ const auto* v=reinterpret_cast<const uint16_t*>(buf.data()); n=buf.size()/2; if(!n)return "empty"; float f=h32(v[0]); mn=mx=f; for(size_t i=0;i<n;i++){float x=h32(v[i]);mn=std::min(mn,x);mx=std::max(mx,x);sum+=x;} }
  else return "non-float";
  std::ostringstream o; o<<"min="<<mn<<" max="<<mx<<" mean="<<(float)(sum/(double)n);
  return o.str();
}

// Run Qwen3 text encoder on demand: ids[512] + att[512] -> cap_feats f32[512,2560]
// (hidden states are NHWC [1,2560,512]; cap_feats expects [512,2560] = transpose)
static bool pipeTextEncoder(Runtime&r,const std::vector<int>& ids,const std::vector<uint8_t>& att,
                            std::vector<float>& capF,std::string& errOut){
  QnnSet&S=r.hp;
  if(!(S.backend&&S.context)){ errOut="HTP backend not initialized"; return false; }
  // embedding input: int64 [1,512]
  std::vector<uint8_t> idsQ(512*8);
  auto* id64=reinterpret_cast<int64_t*>(idsQ.data());
  for(int i=0;i<512;i++) id64[i]=(int64_t)ids[i];
  // attention mask [1,512,512] causal+padding
  std::vector<uint8_t> maskQ(512*512*4);
  { auto* m=reinterpret_cast<float*>(maskQ.data());
    const float negInf=-3.402823466e38f;
    for(int q=0;q<512;q++) for(int k=0;k<512;k++)
      m[q*512+k]=(k>q||!att[k])?negInf:0.0f; }
  std::vector<uint8_t> cosQ(1*128*512*4), sinQ(1*128*512*4);
  if(!readFile(r.root+"/assets/cos_qnn_f32.raw",cosQ)||!readFile(r.root+"/assets/sin_qnn_f32.raw",sinQ)){
    errOut="cos/sin assets missing"; return false; }
  int64_t ms=0;
  // Text cache: embedding + 6 layer groups composed once (like tcache) to
  // avoid repeated compose/free churn that spikes RSS and triggers LMK.
  std::vector<uint8_t> hiddenQ(1*2560*512*4);
  if(!r.textCacheReady){
    if(r.textCacheError.empty()){
      Qnn_ContextHandle_t ctx=nullptr;
      r.textCache.resize(7);
      auto& emb=r.textCache[0]; emb.name="embedding"; emb.libName="libqnn_embedding.so";
      writeProgress(r,"pipeline: 编译 text 段(一次性)",11,100);
      {
        std::string bin; segBinPath(r,"embedding",bin);
        FILE* probe=fopen(bin.c_str(),"rb"); bool haveBin=(probe!=nullptr); if(probe)fclose(probe);
        if(!haveBin){
          if(S.api.contextCreate(S.backend,S.device,nullptr,&ctx)!=QNN_SUCCESS){ r.textCacheError="embedding contextCreate failed"; }
          else if(!composeSegment(r,emb,S,ctx)){ r.textCacheError="embedding compose: "+emb.error; S.api.contextFree(ctx,nullptr); }
          else { cacheSaveBinaryImpl(S,bin,ctx); S.api.contextFree(ctx,nullptr); releaseSegment(r,emb); logline(r,"textcache compile+save embedding"); }
        }
      }
      const char* groups[]={"layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29","layers_30_35"};
      for(int gi=0;gi<6&&r.textCacheError.empty();gi++){
        auto& sg=r.textCache[1+gi];
        sg.name=std::string("text_")+groups[gi];
        std::string bin; segBinPath(r,sg.name,bin);
        FILE* probe=fopen(bin.c_str(),"rb"); bool haveBin=(probe!=nullptr); if(probe)fclose(probe);
        if(haveBin){
          Qnn_ContextHandle_t rc=nullptr; std::string e2;
          // verify binary loads (cheap sanity); then free again - step-restore mode
          if(cacheLoadSegment(r,S,std::string("libqnn_"),sg.name,sg,rc,e2)){
            S.api.contextFree(rc,nullptr); releaseSegment(r,sg);
            r.textCtx.push_back(nullptr);
            logline(r,"textcache binary ok "+sg.name+" (step-restore)");
            continue;
          }
          logline(r,"textcache binary failed "+sg.name+": "+e2+" (recompiling)");
        }
        ctx=nullptr;
        sg.libName=std::string("libqnn_")+groups[gi]+".so";
        if(S.api.contextCreate(S.backend,S.device,nullptr,&ctx)!=QNN_SUCCESS){ r.textCacheError=std::string(groups[gi])+" contextCreate failed"; break; }
        if(!composeSegment(r,sg,S,ctx)){ r.textCacheError=sg.name+" compose: "+sg.error; S.api.contextFree(ctx,nullptr); break; }
        cacheSaveBinaryImpl(S,bin,ctx);
        S.api.contextFree(ctx,nullptr); ctx=nullptr;
        releaseSegment(r,sg);
        r.textCtx.push_back(nullptr);
        logline(r,"textcache compile+save "+sg.name);
      }
      if(r.textCacheError.empty()) r.textCacheReady=true;
      if(r.textCacheError.empty()) r.textCacheReady=true;
      else logline(r,"textcache failed: "+r.textCacheError);
    }
    if(!r.textCacheReady){ errOut=r.textCacheError; return false; }
  }
  // Exec path (step-restore): one segment context alive at a time.
  {
    Qnn_ContextHandle_t rc=nullptr; std::string e2;
    if(!cacheLoadSegment(r,S,std::string("libqnn_"),std::string("embedding"),r.textCache[0],rc,e2)){
      errOut="embedding restore: "+e2; return false; }
    r.textCtx[0]=rc;
    auto& eg=*r.textCache[0].graphs[0];
    std::vector<uint8_t> embedOut(1*512*2560*4);
    bool ok=runGraph(r,S,eg,{{inTensor(eg,"input_ids"),idsQ.data(),idsQ.size()}},
                 {{outTensor(eg,"hidden_states"),embedOut.data(),embedOut.size()}},&ms);
    logline(r,"pipeline embedding ms="+std::to_string(ms));
    releaseSegment(r,r.textCache[0]);
    if(rc&&S.api.contextFree){S.api.contextFree(rc,nullptr);r.textCtx[0]=nullptr;}
    if(!ok){ errOut="embedding exec: "+r.error; return false; }
    { const auto* srcc=reinterpret_cast<const float*>(embedOut.data());
      auto* dst=reinterpret_cast<float*>(hiddenQ.data());
      for(int s=0;s<512;s++) for(int d=0;d<2560;d++) dst[d*512+s]=srcc[s*2560+d]; }
    for(int gi=0;gi<6;gi++){
      int idx=1+gi;
      if(!cacheLoadSegment(r,S,std::string("libqnn_"),r.textCache[idx].name,r.textCache[idx],rc,e2)){
        errOut=r.textCache[idx].name+" restore: "+e2; return false; }
      r.textCtx[idx]=rc;
      auto& g=*r.textCache[idx].graphs[0];
      std::vector<uint8_t> layerOut(1*2560*512*4);
      ok=runGraph(r,S,g,{
            {inTensor(g,"hidden_states"),hiddenQ.data(),hiddenQ.size()},
            {inTensor(g,"attention_mask"),maskQ.data(),maskQ.size()},
            {inTensor(g,"cos"),cosQ.data(),cosQ.size()},
            {inTensor(g,"sin"),sinQ.data(),sinQ.size()}},
          {{outTensor(g,"hidden_states_out"),layerOut.data(),layerOut.size()}},&ms);
      hiddenQ.swap(layerOut);
      releaseSegment(r,r.textCache[idx]);
      if(rc&&S.api.contextFree){S.api.contextFree(rc,nullptr);r.textCtx[idx]=nullptr;}
      if(!ok){ errOut=r.textCache[idx].name+" exec: "+r.error; return false; }
    }
  }
  // hiddenQ is [1,2560,512]; cap_feats wants [512,2560]: take last-but-one row? No:
  // Z-Image conditioning = full sequence [512,2560] (all tokens). Transpose back.
  capF.resize((size_t)512*2560);
  { const auto* src=reinterpret_cast<const float*>(hiddenQ.data());
    for(int s=0;s<512;s++) for(int d=0;d<2560;d++) capF[(size_t)s*2560+d]=src[(size_t)d*512+s]; }
  return true;
}

// VAE decode: latent CHW f32 -> RGB HWC uint8 image
// Last generated image (raw RGB888, 512x512x3). Guarded by mutex; Java copies
// it out then calls nativeClearLastImage to release the buffer.
static std::mutex g_imgMtx;
static std::vector<uint8_t> g_lastImage;
static bool pipeVaeDecode(Runtime&r,const std::vector<float>& latentCHW,
                          std::vector<uint8_t>& rgb,std::string& errOut){
  QnnSet&S=r.hp;
  if(!r.graphs||!r.count||!(S.backend&&S.context)){ errOut="VAE graph not composed: "+r.error; return false; }
  auto&g=*r.graphs[0];
  if(g.numInputTensors!=1||g.numOutputTensors!=1){ errOut="VAE tensor count mismatch"; return false; }
  auto inT=g.inputTensors[0]; auto outT=g.outputTensors[0];
  // pack latent (CHW -> QNN NHWC layout [1,64,64,16])
  std::vector<uint8_t> latentQ; packLatentBuffer(latentCHW.data(),inT.v1.dataType,latentQ);
  std::vector<uint8_t> imgQ(bytes(outT),0);
  int64_t ms=0;
  if(!runGraph(r,S,g,{{&inT,latentQ.data(),latentQ.size()}},{{&outT,imgQ.data(),imgQ.size()}},&ms)){
    errOut="VAE exec: "+r.error; return false; }
  logline(r,"pipeline vae ms="+std::to_string(ms));
  // output image [1,512,512,3] float NHWC -> rgb HWC uint8
  const size_t H=512,W=512,C=3;
  rgb.resize(H*W*C);
  for(size_t h=0;h<H;h++) for(size_t w=0;w<W;w++) for(size_t c=0;c<C;c++){
    float x=elemToF32(imgQ,outT.v1.dataType,(h*W+w)*C+c); // [-1,1]
    float u=(x*0.5f+0.5f)*255.0f; if(u<0)u=0; if(u>255)u=255;
    rgb[(h*W+w)*C+c]=(uint8_t)(u+0.5f);
  }
  return true;
}

// VAE-only health probe: zero latent through the shiftfix VAE. No tokenizer,
// no text encoder, no transformer - cheap startup check.
extern "C" JNIEXPORT jstring JNICALL Java_com_example_zimage_ZImageRuntime_nativeVaeProbe(JNIEnv*e,jobject,jlong h){
  auto*r=rt(h);
  QnnSet&S=r->hp;
  if(!r->graphs||!r->count||!(S.backend&&S.context)) return e->NewStringUTF(("VAE unavailable: "+r->error).c_str());
  auto&g=*r->graphs[0];
  if(g.numInputTensors!=1||g.numOutputTensors!=1) return e->NewStringUTF("VAE tensor count mismatch");
  std::vector<float> latent(16*64*64,0.0f);
  auto inT=g.inputTensors[0]; auto outT=g.outputTensors[0];
  std::vector<uint8_t> latentQ; packLatentBuffer(latent.data(),inT.v1.dataType,latentQ);
  std::vector<uint8_t> imgQ(bytes(outT),0);
  inT.v1.memType=QNN_TENSORMEMTYPE_RAW; inT.v1.clientBuf.data=latentQ.data(); inT.v1.clientBuf.dataSize=latentQ.size();
  outT.v1.memType=QNN_TENSORMEMTYPE_RAW; outT.v1.clientBuf.data=imgQ.data(); outT.v1.clientBuf.dataSize=imgQ.size();
  auto x=S.api.graphExecute(g.graph,&inT,1,&outT,1,nullptr,nullptr);
  if(x!=QNN_SUCCESS) return e->NewStringUTF(("VAE execute="+err(x)).c_str());
  std::ostringstream o;o<<"VAE probe OK: "<<tensorFloatStats(outT,imgQ);
  logline(*r,o.str());
  return e->NewStringUTF(o.str().c_str());
}
// --- utilization monitor (P2): CPU via /proc/stat deltas, GPU via sysfs busy
// counters, NPU best-effort (devfreq freq / remoteproc load). Returns one line:
// "cpu=12.3 gpu=45.6 npu=800MHz" (values may be -1 / "n/a" when unreadable).
static long long g_prevCpuTotal=0,g_prevCpuIdle=0;
static long long g_prevGpuBusy=0,g_prevGpuTotal=0;
static double readCpuPct(){
  FILE* f=fopen("/proc/stat","r");
  if(!f) return -1.0;
  char tag[16]={0}; long long u,n,s,idle,io,irq,sirq,st;
  int got=fscanf(f,"%15s %lld %lld %lld %lld %lld %lld %lld %lld",tag,&u,&n,&s,&idle,&io,&irq,&sirq,&st);
  fclose(f);
  if(got<5||std::string(tag)!="cpu") return -1.0;
  long long total=u+n+s+idle+io+irq+sirq+st;
  double pct=-1.0;
  if(g_prevCpuTotal>0){
    long long dT=total-g_prevCpuTotal, dI=idle-g_prevCpuIdle;
    if(dT>0) pct=100.0*(double)(dT-dI)/(double)dT;
  }
  g_prevCpuTotal=total; g_prevCpuIdle=idle;
  return pct<0?0.0:pct;
}
static double readGpuBusyPct(){
  // Adreno exports "busy total" (monotonic usec counters) at either path.
  FILE* f=nullptr;
  const char* paths[]={"/sys/kernel/gpu/gpu_busy","/sys/class/kgsl/kgsl-3d0/gpubusy"};
  for(auto p:paths){ f=fopen(p,"r"); if(f) break; }
  if(!f) return -1.0;
  long long busy=0,total=0;
  int got=fscanf(f,"%lld %lld",&busy,&total);
  fclose(f);
  if(got!=2||total<=0) return -1.0;
  double pct=-1.0;
  if(g_prevGpuBusy>=0&&g_prevGpuTotal>0){
    long long dB=busy-g_prevGpuBusy,dT=total-g_prevGpuTotal;
    if(dT>0) pct=100.0*(double)dB/(double)dT;
  }
  g_prevGpuBusy=busy; g_prevGpuTotal=total;
  return pct<0?0.0:pct;
}
static std::string readNpu(){
  // Hexagon HTP: no standard busy node; report cdsp remoteproc load/freq best-effort.
  FILE* f=fopen("/sys/class/devfreq/32300000.remoteproc-cdsp/cur_freq","r");
  if(f){ long long khz=0; int got=fscanf(f,"%lld",&khz); fclose(f);
    if(got==1) return std::to_string(khz/1000)+"MHz"; }
  f=fopen("/sys/devices/platform/soc/32300000.remoteproc-cdsp/remoteproc/remoteproc*","r");
  if(f) fclose(f);
  return "n/a";
}
static std::string readSysLoad(){
  FILE* f=fopen("/proc/loadavg","r");
  if(f){ char b[64]={0}; char* g=fgets(b,sizeof b,f); fclose(f);
    if(g){ char* sp=strchr(b,' '); if(sp) *sp=0; return b; } }
  return "n/a";
}
extern "C" JNIEXPORT jstring JNICALL Java_com_example_zimage_ZImageRuntime_nativePollUtilization(JNIEnv*e,jobject,jlong){
  double cpu=readCpuPct(), gpu=readGpuBusyPct();
  std::ostringstream o;
  o.precision(1); o<<std::fixed;
  o<<"cpu="<<(cpu<0?0.0:cpu)<<" gpu="<<(gpu<0?-1.0:gpu)<<" npu="<<readNpu();
  return e->NewStringUTF(o.str().c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_com_example_zimage_ZImageRuntime_nativeGenerate(JNIEnv*e,jobject,jlong h,jstring jPrompt,jint jWidth,jint jHeight,jint jSteps,jlong jSeed){
  auto*r=rt(h);
  if(!r) return e->NewStringUTF("runtime handle invalid (destroyed?)");
  const char*pj=e->GetStringUTFChars(jPrompt,nullptr);
  std::string prompt=pj?pj:"a red paper lantern"; if(pj)e->ReleaseStringUTFChars(jPrompt,pj);
  const int steps=(jSteps>0&&jSteps<=16)?jSteps:8;
  // Z-Image latent is fixed 64x64 -> output 512x512. Honor requested dims only
  // for the PPM header (UI still passes 512) so we never lie about resolution.
  const int W=(jWidth==512)?512:512; const int H=(jHeight==512)?512:512;
  (void)W;(void)H;
  auto t0=std::chrono::steady_clock::now();
  writeProgress(*r,"pipeline: tokenize",5,100);
  zpipe::BpeModel bm;
  if(!bm.load(r->root+"/assets")){ logline(*r,"tokenizer load failed"); return e->NewStringUTF("tokenizer data missing under assets/"); }
  std::vector<int> ids; std::vector<uint8_t> att;
  zpipe::encodePrompt(bm,prompt,ids,att);
  { std::ostringstream d; d<<"tokenize: first12=";
    for(int i=0;i<12&&i<512;i++) d<<ids[i]<<" ";
    int n=0; for(auto m:att) n+=m?1:0;
    d<<" maskSum="<<n; logline(*r,d.str()); }

  writeProgress(*r,"pipeline: text encoder",10,100);
  std::vector<float> capF; std::string errOut;
  if(!pipeTextEncoder(*r,ids,att,capF,errOut)){ logline(*r,"text encoder failed: "+errOut); return e->NewStringUTF(("text encoder failed: "+errOut).c_str()); }
  logline(*r,"cap_feats "+f32Stats(std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(capF.data()),reinterpret_cast<const uint8_t*>(capF.data())+capF.size()*4)));

  // scheduler setup: shift=3.0, steps from UI
  const double shift=3.0;
  std::vector<double> sigmas; zpipe::buildSigmas(steps,shift,sigmas);
  std::vector<float> latent(16*64*64);
  zpipe::fillRandn(latent,(uint64_t)jSeed);
  auto tDenoise0=std::chrono::steady_clock::now(); double transformerMs=0;
  for(int i=0;i<steps;i++){
    float tNorm=(float)(1.0-sigmas[i]);       // model input t = (1000-t)/1000 = 1-sigma
    double dt=sigmas[i+1]-sigmas[i];
    std::vector<float> noise;
    if(!pipeTransformer(*r,latent,tNorm,capF,att,noise,errOut)){
      logline(*r,"transformer failed step "+std::to_string(i)+": "+errOut);
      return e->NewStringUTF(("transformer failed: "+errOut).c_str()); }
    // euler: x <- x + dt * v where v = -noise_pred (flow matching velocity)
    for(size_t k=0;k<latent.size();k++) latent[k]+= (float)(dt*(-noise[k]));
    writeProgress(*r,std::string("pipeline: denoise ")+std::to_string(i+1)+"/"+std::to_string(steps),20+(i+1)*60/steps,100);
  }
  transformerMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tDenoise0).count();

  writeProgress(*r,"pipeline: VAE decode",85,100);
  // NOTE: libqnn_vae_shiftfix.so already contains latents/0.3611+0.1159 inside
  // the graph (export_vae_onnx.py "shiftfix"); feed RAW scheduler output.
  std::vector<uint8_t> img;
  if(!pipeVaeDecode(*r,latent,img,errOut)){ logline(*r,"VAE failed: "+errOut); return e->NewStringUTF(("VAE failed: "+errOut).c_str()); }
  { std::lock_guard<std::mutex> lk(g_imgMtx); g_lastImage=img; }
  double totalMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
  writeProgress(*r,"pipeline 完成",100,100);
  // Save PPM (P6, 512x512 RGB) so the image survives the JNI return.
  {
    std::string pp=r->root+"/output.ppm";
    FILE* fp=fopen(pp.c_str(),"wb");
    if(fp){
      fprintf(fp,"P6\n512 512\n255\n");
      fwrite(img.data(),1,img.size(),fp);
      fclose(fp);
      logline(*r,"saved "+pp+" ("+std::to_string(img.size())+" bytes)");
    } else logline(*r,"WARN: cannot write "+pp);
  }
  std::ostringstream o;o<<"generated "<<img.size()<<" bytes in "<<totalMs<<" ms (transformer "<<transformerMs<<" ms)";
  logline(*r,o.str());
  return e->NewStringUTF(o.str().c_str());
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_com_example_zimage_ZImageRuntime_nativeGetLastImage(JNIEnv*e,jobject,jlong h){
  std::lock_guard<std::mutex> lk(g_imgMtx);
  if(g_lastImage.empty()) return nullptr;
  jbyteArray arr=e->NewByteArray((jsize)g_lastImage.size());
  e->SetByteArrayRegion(arr,0,(jsize)g_lastImage.size(),(const jbyte*)g_lastImage.data());
  return arr;
}
extern "C" JNIEXPORT void JNICALL Java_com_example_zimage_ZImageRuntime_nativeClearLastImage(JNIEnv*,jobject,jlong){
  std::lock_guard<std::mutex> lk(g_imgMtx);
  std::vector<uint8_t>().swap(g_lastImage);
}
extern "C" JNIEXPORT void JNICALL Java_com_example_zimage_ZImageRuntime_nativeDestroy(JNIEnv*,jobject,jlong h){auto*r=reinterpret_cast<Runtime*>(h);if(!r||r->magic!=0x5A494D47)return;r->magic=0;if(r->graphs&&r->freeGraphs)r->freeGraphs(&r->graphs,r->count);for(auto&sg:r->segments)releaseSegment(*r,sg);for(auto&sg:r->textSegments)releaseSegment(*r,sg);for(auto&sg:r->tcache)releaseSegment(*r,sg);for(auto&sg:r->textCache)releaseSegment(*r,sg);{QnnSet&S=r->main;for(auto ctx:r->tcCtx)if(ctx&&S.api.contextFree)S.api.contextFree(ctx,nullptr);}r->main.release(r);r->hp.release(r);if(r->model)dlclose(r->model);if(r->vae)dlclose(r->vae);if(r->htp)dlclose(r->htp);if(r->gpu)dlclose(r->gpu);if(r->sys)dlclose(r->sys);delete r;}
