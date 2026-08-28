# Third-Party Notices

This project distributes an APK (`app-debug.apk`) that incorporates third-party
components. The notices below satisfy attribution and notice requirements of
the respective licenses. **Do not remove or alter these notices when
redistributing the APK.**

> **Project disclaimer**: All code in this project was produced through
> vibe coding (AI-assisted development) and is **not actively maintained**.
> Provided as-is for reference only — use at your own risk.

---

## Qualcomm AI Runtime (QAIRT) — Proprietary

- **Component**: `lib/arm64-v8a/libQnnHtpV79Stub.so` (in the APK)
- **Source**: Qualcomm AI Runtime (QAIRT) SDK 2.49.0, file
  `lib/aarch64-android/libQairtHtpV79Stub.so` (renamed on packaging)
- **License**: Qualcomm AI Stack License — **proprietary / confidential**.
  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
  All rights reserved. Confidential and Proprietary - Qualcomm Technologies, Inc.

This component is distributed **in object code form, solely as incorporated in
this application**, as permitted by Section 1 (Grant of License) of the
Qualcomm AI Stack License. It is **not** licensed for standalone
distribution/sublicensing.

Notable obligations (summary — see the full license for details):

- No reverse engineering, disassembly, or decompilation of the component.
- Keep all proprietary notices intact.
- **Export control (Section 10(f))**: do not export or make available to
  embargoed countries/territories (currently incl. Cuba, Iran, North Korea,
  Syria, Crimea, Donetsk/Luhansk regions of Ukraine) or restricted parties
  without prior U.S. government authorization.
- You indemnify QTI per Section 9 for claims arising from your use/distribution.

Full license text ships with the QAIRT SDK as `LICENSE.pdf`; third-party
notices within it are in `NOTICE.txt`.

---

## Google AI Edge LiteRT (TensorFlow Lite) — Apache 2.0

- **Component**: `lib/arm64-v8a/libtensorflowlite_jni.so`,
  `libtensorflowlite_gpu_jni.so` (and other ABIs) in the APK
- **Source**: `com.google.ai.edge.litert:litert:1.4.2` /
  `litert-gpu:1.4.2` (Gradle dependency)
- **License**: Apache License 2.0 — https://www.apache.org/licenses/LICENSE-2.0

## Google Material Components / AndroidX — Apache 2.0

- **Components**: `com.google.android.material:material:1.12.0`,
  `androidx.appcompat:appcompat:1.7.0`,
  `androidx.activity:activity-ktx:1.9.3`,
  `androidx.recyclerview:recyclerview:1.3.2`,
  `androidx.core:core-ktx:1.13.1`
- **License**: Apache License 2.0 — https://www.apache.org/licenses/LICENSE-2.0

## Qwen Tokenizer Data — Apache 2.0

- **Components**: `assets/qwen_vocab.tsv`, `assets/qwen_merges.txt` (in the APK)
- **Source**: Qwen tokenizer (Alibaba), part of the Qwen model family
- **License**: Apache License 2.0 — https://www.apache.org/licenses/LICENSE-2.0

---

## Project's own code

- `libzimage_runtime.so` and all Kotlin sources are original work of this
  project. License: see `LICENSE` (if present) or contact the author.

---

## Notes

- **Model weights** (Z-Image-Turbo transformer/text/VAE QNN artifacts) are
  **not** packaged in the APK and are not distributed via this repository.
  They are governed by the ModelScope/Tongyi-MAI model license.
- This notice file must be redistributed together with the APK.
