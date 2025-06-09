import SwiftUI
import AppKit

enum OverlayState {
    case idle, listening, transcribing(String), done(String)
}

struct OverlayView: View {
    @State private var state: OverlayState = .idle
    @State private var recorder = Recorder()
    @State private var modelURL: URL? = nil
    @State private var manager = ModelManager()
    @State private var task: Task<Void, Never>? = nil

    var body: some View {
        VStack {
            switch state {
            case .idle:
                Image(systemName: "mic")
                    .onTapGesture { toggleListening() }
            case .listening:
                ProgressView("Listening…")
                    .onAppear { startRecording() }
            case .transcribing(let text):
                ProgressView(text)
            case .done(let text):
                Text(text)
            }
        }
        .frame(width: 320, height: 200)
        .background(Material.thick)
        .cornerRadius(12)
        .onReceive(NotificationCenter.default.publisher(for: .toggleOverlay)) { _ in
            toggleListening()
        }
    }

    private func toggleListening() {
        print("toggleListening called, current state: \(state)")

        switch state {
        case .idle:
            state = .listening
            print("State changed to listening")
        case .listening:
            print("Stopping recording")

            stopRecording()
        default:
            print("State is: \(state)")

            break
        }
    }

    private func startRecording() {
        print("startRecording() called")

        task = Task {
            do {
                try await manager.ensureModel()
                let file = try FileManager.default
                    .temporaryDirectory.appending(path: "record.wav")
                try await recorder.startRecording(toOutputFile: file, delegate: nil)
            } catch {
                print("ERROR in startRecording: \(error)")
                            // Also update the UI to show the error
                            DispatchQueue.main.async {
                                self.state = .done("Error: \(error.localizedDescription)")
                            }
            }
        }
    }

    private func stopRecording() {
        print("stopRecording() called")

        task?.cancel()
        Task {
            print("Stopping recorder...")

            recorder.stopRecording()
            if let url = recorder.currentFile {
                state = .transcribing("Transcribing…")
                let ctx = try? WhisperContext.createContext(path: manager.modelPath().path)
                print("Whisper context created: \(ctx != nil)")

                if let data = try? decodeWaveFile(url) {
                    print("Audio data decoded, samples: \(data.count)")

                    ctx?.fullTranscribe(samples: data, language: "")
                    let text = ctx?.getTranscription() ?? ""
                    print("Transcription result: '\(text)'")

                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                    state = .done(text)
                }
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                state = .idle
            }
        }
    }
}

extension Notification.Name {
    static let toggleOverlay = Notification.Name("ToggleOverlay")
}
