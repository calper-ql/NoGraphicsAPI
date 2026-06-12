// Cocoa + CAMetalLayer windowing backend for the Metal samples.
// Provides the same ngapi:: interface as window_glfw.cpp / window_sdl3.cpp.
// createSurface() passes the CAMetalLayer pointer (as void*) to
// gpuCreateSurface(), which the Metal backend stores in GpuSurface_T.

#define GPU_EXPOSE_INTERNAL
#include "window.h"

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <array>
#include <cstddef>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Close delegate
// ---------------------------------------------------------------------------

@interface NgapiWindowDelegate : NSObject <NSWindowDelegate>
- (instancetype)initWithFlag:(bool*)closeFlag;
@end

@implementation NgapiWindowDelegate {
    bool* _closeFlag;
}
- (instancetype)initWithFlag:(bool*)closeFlag {
    self = [super init];
    if (self) _closeFlag = closeFlag;
    return self;
}
- (BOOL)windowShouldClose:(NSWindow*)sender {
    *_closeFlag = true;
    return YES;
}
@end

// ---------------------------------------------------------------------------
// NSApp one-time init (idempotent)
// ---------------------------------------------------------------------------

static void ngapi_initNSApp()
{
    static bool done = false;
    if (done) return;
    done = true;

    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];
    [NSApp finishLaunching];
}

// ---------------------------------------------------------------------------
// Virtual key codes (HIToolbox/Events.h constants, inlined to avoid Carbon dep)
// ---------------------------------------------------------------------------
enum : uint16_t {
    kVK_ANSI_A    = 0x00, kVK_ANSI_S = 0x01, kVK_ANSI_T = 0x11,
    kVK_ANSI_R    = 0x0F, kVK_ANSI_X = 0x07,
    kVK_Escape    = 0x35,
    kVK_UpArrow   = 0x7E, kVK_DownArrow  = 0x7D,
    kVK_LeftArrow = 0x7B, kVK_RightArrow = 0x7C,
};

// ---------------------------------------------------------------------------
// ngapi namespace implementation
// ---------------------------------------------------------------------------

namespace ngapi
{
    struct Window
    {
        NSWindow*            nsWindow  = nil;
        CAMetalLayer*        layer     = nil;
        NgapiWindowDelegate* delegate  = nil;
        bool                 closeRequested = false;
        std::array<bool, static_cast<size_t>(Key::Count)> down{};
        std::array<bool, static_cast<size_t>(Key::Count)> pressed{};

        // NGAPI_MAX_FRAMES env var: request close after N pollEvents calls
        // (headless/CI runs of the windowed samples). -1 = no cap.
        long maxFrames  = -1;
        long frameCount = 0;
    };

    static Key keyFromCode(uint16_t code)
    {
        switch (code)
        {
        case kVK_ANSI_A:    return Key::A;
        case kVK_ANSI_S:    return Key::S;
        case kVK_ANSI_T:    return Key::T;
        case kVK_ANSI_R:    return Key::R;
        case kVK_ANSI_X:    return Key::X;
        case kVK_Escape:    return Key::Escape;
        case kVK_UpArrow:   return Key::Up;
        case kVK_DownArrow: return Key::Down;
        case kVK_LeftArrow: return Key::Left;
        case kVK_RightArrow:return Key::Right;
        default:            return Key::Count; // sentinel: unknown
        }
    }

    Window* createWindow(const char* title, int width, int height)
    {
        ngapi_initNSApp();

        NSRect frame = NSMakeRect(0, 0, (CGFloat)width, (CGFloat)height);
        NSWindowStyleMask style =
            NSWindowStyleMaskTitled |
            NSWindowStyleMaskClosable |
            NSWindowStyleMaskMiniaturizable;

        NSWindow* nsw = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:style
                        backing:NSBackingStoreBuffered
                          defer:NO];
        [nsw setTitle:[NSString stringWithUTF8String:title]];
        [nsw center];

        // Embed a CAMetalLayer in the content view.
        NSView* view = [[NSView alloc] initWithFrame:frame];
        view.wantsLayer = YES;

        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.device        = MTLCreateSystemDefaultDevice();
        layer.pixelFormat   = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = NO; // allow compute writes to drawable texture
        layer.drawableSize  = CGSizeMake((CGFloat)width, (CGFloat)height);
        view.layer = layer;

        [nsw setContentView:view];
        [nsw makeKeyAndOrderFront:nil];

        Window* w       = new Window();
        w->nsWindow     = nsw;
        w->layer        = layer;
        w->delegate     = [[NgapiWindowDelegate alloc] initWithFlag:&w->closeRequested];
        [nsw setDelegate:w->delegate];
        if (const char* s = getenv("NGAPI_MAX_FRAMES"))
            w->maxFrames = atol(s);
        return w;
    }

    void destroyWindow(Window* window)
    {
        if (!window) return;
        [window->nsWindow setDelegate:nil];
        [window->nsWindow close];
        delete window;
    }

    GpuSurface createSurface(Window* window)
    {
        // Pass the CAMetalLayer as void* — the Metal backend stores it in GpuSurface_T.
        return gpuCreateSurface((__bridge void*)window->layer);
    }

    void destroySurface(Window* /*window*/, GpuSurface surface)
    {
        gpuDestroySurface(surface);
    }

    void pollEvents(Window* window)
    {
        // Reset per-frame pressed state.
        window->pressed.fill(false);

        NSEvent* event;
        while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                           untilDate:nil
                                              inMode:NSDefaultRunLoopMode
                                             dequeue:YES]))
        {
            if (event.type == NSEventTypeKeyDown || event.type == NSEventTypeKeyUp)
            {
                bool isDown = (event.type == NSEventTypeKeyDown);
                Key  k      = keyFromCode(event.keyCode);
                if (k != Key::Count)
                {
                    size_t idx         = static_cast<size_t>(k);
                    window->pressed[idx] = isDown && !window->down[idx];
                    window->down[idx]    = isDown;
                }
                if (event.keyCode == kVK_Escape && isDown)
                    window->closeRequested = true;
            }
            [NSApp sendEvent:event];
        }

        if (window->maxFrames >= 0 && ++window->frameCount >= window->maxFrames)
            window->closeRequested = true;
    }

    bool shouldClose(Window* window)
    {
        return window->closeRequested;
    }

    bool isKeyDown(Window* window, Key key)
    {
        return window->down[static_cast<size_t>(key)];
    }

    bool wasKeyPressed(Window* window, Key key)
    {
        return window->pressed[static_cast<size_t>(key)];
    }

} // namespace ngapi
