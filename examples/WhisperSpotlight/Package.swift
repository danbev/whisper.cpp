// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "WhisperSpotlight",
    platforms: [.macOS("15")],
    products: [
        .executable(name: "WhisperSpotlight", targets: ["WhisperSpotlight"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(name: "WhisperSpotlight", dependencies: ["whisper"], path: "", exclude: ["Tests"]),
        .binaryTarget(name: "whisper", path: "../../build-apple/whisper.xcframework"),
        .testTarget(name: "WhisperSpotlightTests", dependencies: ["WhisperSpotlight"], path: "Tests")
    ]
)
