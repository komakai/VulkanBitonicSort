package com.android.ndk.samples.buildlogic

import org.gradle.api.JavaVersion

object Versions {
    const val COMPILE_SDK = 34
    const val TARGET_SDK = 34
    const val MIN_SDK = 21
    const val NDK = "27.0.11718014" // r27b
    const val CMAKE = "3.22.1"
    val JAVA = JavaVersion.VERSION_1_8
}
