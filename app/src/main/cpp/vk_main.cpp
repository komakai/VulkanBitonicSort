/*
 * Based on https://gist.github.com/sheredom/523f02bbad2ae397d7ed255f3f3b5a7f and
 * https://github.com/android/ndk-samples/tree/main/hello-vulkan
 */

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "hellovk.h"

/*
 * Shared state for the app. This will be accessed within lifecycle callbacks
 * such as APP_CMD_START or APP_CMD_INIT_WINDOW.
 *
 * We store:
 * struct android_app - a pointer to the Android application handle
 *
 * vkt::HelloVK - a pointer to our (this) Vulkan application in order to call
 *  the rendering logic
 *
 * bool canCompute - a flag which signals that we are ready to call the vulkan
 * rendering logic
 *
 */
struct VulkanEngine {
    struct android_app *app;
    vkt::HelloVK *app_backend;
    bool canCompute = false;
};

/**
 * Called by the Android runtime whenever events happen so the
 * app can react to it.
 */
static void HandleCmd(struct android_app *app, int32_t cmd) {
    auto *engine = (VulkanEngine *)app->userData;
    switch (cmd) {
        case APP_CMD_START:
            if (engine->app->window != nullptr) {
                engine->app_backend->reset(app->window, app->activity->assetManager);
                engine->app_backend->initVulkanCompute();
                engine->canCompute = true;
            }
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            LOGI("Called - APP_CMD_INIT_WINDOW");
            if (engine->app->window != nullptr) {
                LOGI("Setting a new surface");
                engine->app_backend->reset(app->window, app->activity->assetManager);
                if (!engine->app_backend->initialized) {
                    LOGI("Starting application");
                    engine->app_backend->initVulkanCompute();
                }
                engine->canCompute = true;
            }
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            engine->canCompute = false;
            break;
        case APP_CMD_DESTROY:
            // The window is being hidden or closed, clean it up.
            LOGI("Destroying");
            engine->app_backend->cleanupCompute();
        default:
            break;
    }
}

/*
 * Key events filter to GameActivity's android_native_app_glue. This sample does
 * not use/process any input events, return false for all input events so system
 * can still process them.
 */
extern "C" bool VulkanKeyEventFilter(const GameActivityKeyEvent *event) {
    return false;
}
extern "C" bool VulkanMotionEventFilter(const GameActivityMotionEvent *event) {
    return false;
}

/*
 * Process user touch and key events. GameActivity double buffers those events,
 * applications can process at any time. All of the buffered events have been
 * reported "handled" to OS. For details, refer to:
 * d.android.com/games/agdk/game-activity/get-started#handle-events
 */
static void HandleInputEvents(struct android_app *app) {
    auto inputBuf = android_app_swap_input_buffers(app);
    if (inputBuf == nullptr) {
        return;
    }

    // For the minimum, apps need to process the exit event (for example,
    // listening to AKEYCODE_BACK). This sample has done that in the Kotlin side
    // and not processing other input events, we just reset the event counter
    // inside the android_input_buffer to keep app glue code in a working state.
    android_app_clear_motion_events(inputBuf);
    android_app_clear_motion_events(inputBuf);
}

/*
 * Entry point required by the Android Glue library.
 * This can also be achieved more verbosely by manually declaring JNI functions
 * and calling them from the Android application layer.
 */
void android_main(struct android_app *state) {
    VulkanEngine engine{};
    vkt::HelloVK vulkanBackend{};

    engine.app = state;
    engine.app_backend = &vulkanBackend;
    state->userData = &engine;
    state->onAppCmd = HandleCmd;

    android_app_set_key_event_filter(state, VulkanKeyEventFilter);
    android_app_set_motion_event_filter(state, VulkanMotionEventFilter);

    while (!state->destroyRequested) {
        android_poll_source *source;
        auto result = ALooper_pollOnce(engine.canCompute ? 0 : -1, nullptr, nullptr,
                                       (void **)&source);
        if (result == ALOOPER_POLL_ERROR) {
            LOGE("ALooper_pollOnce returned an error");
            std::abort();
        }

        if (source != nullptr) {
            source->process(state, source);
        }

        HandleInputEvents(state);

        if (engine.canCompute) {
            engine.app_backend->compute();
        }
    }
}