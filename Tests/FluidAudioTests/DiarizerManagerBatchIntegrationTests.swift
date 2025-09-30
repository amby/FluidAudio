import XCTest
@testable import FluidAudio

/// Tests for the updated DiarizerManager integration with new batch assignment functionality.
final class DiarizerManagerBatchIntegrationTests: XCTestCase {
    
    var diarizerManager: DiarizerManager!
    
    override func setUp() {
        super.setUp()
        let config = DiarizerConfig(
            clusteringThreshold: 0.65,
            minSpeechDuration: 1.0,
            minEmbeddingUpdateDuration: 2.0,
            minActiveFramesCount: 10
        )
        diarizerManager = DiarizerManager(config: config)
    }
    
    override func tearDown() {
        diarizerManager = nil
        super.tearDown()
    }
    
    // MARK: - Basic Integration Tests
    
    func testDiarizerManagerInitialization() {
        XCTAssertNotNil(diarizerManager)
        XCTAssertNotNil(diarizerManager.speakerManager)
    }
    
    func testDiarizerManagerWithCustomConfig() {
        let customConfig = DiarizerConfig(
            clusteringThreshold: 0.7,
            minSpeechDuration: 0.5,
            minEmbeddingUpdateDuration: 1.0,
            minActiveFramesCount: 5
        )
        
        let customDiarizer = DiarizerManager(config: customConfig)
        XCTAssertNotNil(customDiarizer)
        XCTAssertEqual(customDiarizer.config.minActiveFramesCount, 5)
        XCTAssertEqual(customDiarizer.config.clusteringThreshold, 0.7)
    }
    
    // MARK: - SpeakerManager Integration Tests
    
    func testDiarizerManagerSpeakerManagerIntegration() {
        // Test that the DiarizerManager properly integrates with SpeakerManager
        XCTAssertNotNil(diarizerManager.speakerManager)
        
        // Test that we can initialize known speakers
        let testSpeaker = Speaker(
            id: "test-speaker",
            name: "Test Speaker",
            currentEmbedding: [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
            duration: 5.0,
            createdAt: Date(),
            updatedAt: Date()
        )
        
        diarizerManager.initializeKnownSpeakers([testSpeaker])
        
        // Verify the speaker was added by checking that we can validate embeddings
        let testEmbedding: [Float] = [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        XCTAssertTrue(diarizerManager.validateEmbedding(testEmbedding))
    }
    
    func testDiarizerManagerWithMultipleKnownSpeakers() {
        let speakers = [
            Speaker(
                id: "speaker-1",
                name: "Speaker 1",
                currentEmbedding: [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
                duration: 5.0,
                createdAt: Date(),
                updatedAt: Date()
            ),
            Speaker(
                id: "speaker-2",
                name: "Speaker 2",
                currentEmbedding: [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
                duration: 3.0,
                createdAt: Date(),
                updatedAt: Date()
            )
        ]
        
        diarizerManager.initializeKnownSpeakers(speakers)
        
        // Verify both speakers were added by checking that we can validate embeddings
        let testEmbedding1: [Float] = [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        let testEmbedding2: [Float] = [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        XCTAssertTrue(diarizerManager.validateEmbedding(testEmbedding1))
        XCTAssertTrue(diarizerManager.validateEmbedding(testEmbedding2))
    }
    
    // MARK: - Embedding Validation Tests
    
    func testDiarizerManagerEmbeddingValidation() {
        let validEmbedding: [Float] = [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        let invalidEmbedding: [Float] = [0.01, 0.01, 0.01] // Low magnitude
        
        XCTAssertTrue(diarizerManager.validateEmbedding(validEmbedding))
        XCTAssertFalse(diarizerManager.validateEmbedding(invalidEmbedding))
    }
    
    func testDiarizerManagerEmbeddingValidationWithEmptyEmbedding() {
        let emptyEmbedding: [Float] = []
        
        XCTAssertFalse(diarizerManager.validateEmbedding(emptyEmbedding))
    }
    
    func testDiarizerManagerEmbeddingValidationWithNaNEmbedding() {
        let nanEmbedding = Array(repeating: Float.nan, count: 16)
        
        XCTAssertFalse(diarizerManager.validateEmbedding(nanEmbedding))
    }
    
    // MARK: - Audio Validation Tests
    
    func testDiarizerManagerAudioValidation() {
        let validSamples = Array(repeating: Float(0.1), count: 16000) // 1 second at 16kHz
        let result = diarizerManager.validateAudio(validSamples)
        
        XCTAssertTrue(result.isValid)
        XCTAssertEqual(result.durationSeconds, 1.0, accuracy: 0.001)
        XCTAssertTrue(result.issues.isEmpty)
    }
    
    func testDiarizerManagerAudioValidationWithEmptyAudio() {
        let emptySamples: [Float] = []
        let result = diarizerManager.validateAudio(emptySamples)
        
        XCTAssertFalse(result.isValid)
        XCTAssertEqual(result.durationSeconds, 0.0, accuracy: 0.001)
        XCTAssertFalse(result.issues.isEmpty)
    }
    
    func testDiarizerManagerAudioValidationWithShortAudio() {
        let shortSamples = Array(repeating: Float(0.1), count: 100) // Very short
        let result = diarizerManager.validateAudio(shortSamples)
        
        XCTAssertFalse(result.isValid)
        XCTAssertLessThan(result.durationSeconds, 0.1)
        XCTAssertFalse(result.issues.isEmpty)
    }
    
    // MARK: - Configuration Tests
    
    func testDiarizerManagerDefaultConfiguration() {
        let defaultDiarizer = DiarizerManager(config: .default)
        XCTAssertNotNil(defaultDiarizer)
        XCTAssertEqual(defaultDiarizer.config.clusteringThreshold, 0.7)
        XCTAssertEqual(defaultDiarizer.config.minSpeechDuration, 1.0)
    }
    
    func testDiarizerManagerConfigurationValues() {
        XCTAssertEqual(diarizerManager.config.clusteringThreshold, 0.65)
        XCTAssertEqual(diarizerManager.config.minSpeechDuration, 1.0)
        XCTAssertEqual(diarizerManager.config.minEmbeddingUpdateDuration, 2.0)
        XCTAssertEqual(diarizerManager.config.minActiveFramesCount, 10)
    }
    
    // MARK: - Cleanup Tests
    
    func testDiarizerManagerCleanup() {
        // Initialize some speakers first
        let testSpeaker = Speaker(
            id: "cleanup-test",
            name: "Cleanup Test",
            currentEmbedding: [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
            duration: 5.0,
            createdAt: Date(),
            updatedAt: Date()
        )
        
        diarizerManager.initializeKnownSpeakers([testSpeaker])
        
        // Cleanup should not affect the speaker database
        diarizerManager.cleanup()
        // Verify cleanup doesn't break functionality by testing embedding validation
        let testEmbedding: [Float] = [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        XCTAssertTrue(diarizerManager.validateEmbedding(testEmbedding))
    }
    
    // MARK: - Performance Tests
    
    func testDiarizerManagerInitializationPerformance() {
        measure {
            let config = DiarizerConfig()
            let _ = DiarizerManager(config: config)
        }
    }
    
    func testDiarizerManagerSpeakerInitializationPerformance() {
        let speakers = (0..<100).map { i in
            Speaker(
                id: "speaker-\(i)",
                name: "Speaker \(i)",
                currentEmbedding: [Float(i), 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
                duration: Float(i),
                createdAt: Date(),
                updatedAt: Date()
            )
        }
        
        measure {
            diarizerManager.initializeKnownSpeakers(speakers)
        }
    }
}