import SwiftUI
import Carbon

@main
struct WhisperSpotlightApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        WindowGroup {
            OverlayView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .windowLevel(.floating)  // This makes it float above other windows
        .defaultPosition(.center)  // Centers it on screen
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    private var hotkey: GlobalHotkey?
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("ready....")
        //hotkey = GlobalHotkey(keyCode: kVK_Space, modifiers: optionKey)
        hotkey = GlobalHotkey(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey))

        hotkey?.handler = {
            NotificationCenter.default.post(name: .toggleOverlay, object: nil)
        }
    }
}
