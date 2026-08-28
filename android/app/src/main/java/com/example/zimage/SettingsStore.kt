package com.example.zimage

import android.content.Context
import android.content.SharedPreferences

/** Generation settings persisted in SharedPreferences. */
class SettingsStore(context: Context) {
    private val sp: SharedPreferences =
        context.getSharedPreferences("zimage_settings", Context.MODE_PRIVATE)

    // steps: 1..16 (Turbo 模型推荐 4/8)
    var steps: Int
        get() = sp.getInt("steps", 8).coerceIn(1, 16)
        set(v) { sp.edit().putInt("steps", v).apply() }

    // randomSeed: true = seed 0 (random_device), false = fixed seed below
    var randomSeed: Boolean
        get() = sp.getBoolean("random_seed", false)
        set(v) { sp.edit().putBoolean("random_seed", v).apply() }

    var fixedSeed: Long
        get() = sp.getLong("fixed_seed", 42L)
        set(v) { sp.edit().putLong("fixed_seed", v).apply() }

    /** Effective seed to pass to nativeGenerate. */
    val effectiveSeed: Long
        get() = if (randomSeed) 0L else fixedSeed

    // Resolution: Z-Image model latent is fixed 64x64 -> output 512x512.
    // Exposed as a setting for future model variants (e.g. 1024 or 768 via
    // tiled VAE); today only 512 is honored by the C++ side.
    var resolutionLabel: String
        get() = sp.getString("resolution", "512×512") ?: "512×512"
        set(v) { sp.edit().putString("resolution", v).apply() }
}
