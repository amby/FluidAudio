import AVFoundation
import Foundation
import OSLog

/// A high-level streaming ASR manager that provides a simple API for real-time transcription
/// Similar to Apple's SpeechAnalyzer, it handles audio conversion and buffering automatically
@available(macOS 13.0, iOS 16.0, *)
public actor StreamingAsrManager {
    private let logger = AppLogger(category: "StreamingASR")
    private let audioConverter: AudioConverter = AudioConverter()
    private let config: StreamingAsrConfig

    // Audio input stream
    private let inputSequence: AsyncStream<AVAudioPCMBuffer>
    private let inputBuilder: AsyncStream<AVAudioPCMBuffer>.Continuation

    // Transcription output stream
    private var updateContinuation: AsyncStream<StreamingTranscriptionUpdate>.Continuation?

    // ASR components
    private var asrManager: AsrManager?
    private var recognizerTask: Task<Void, Error>?
    private var audioSource: AudioSource = .microphone

    // Sliding window state
    private var segmentIndex: Int = 0
    private var lastProcessedFrame: Int = 0
    private var accumulatedTokenTimings: [TokenTiming] = []  // Accumulated token timings with absolute timestamps

    // Raw sample buffer for sliding-window assembly (absolute indexing)
    private var sampleBuffer: [Float] = []
    private var bufferStartIndex: Int = 0  // absolute index of sampleBuffer[0]
    private var nextWindowCenterStart: Int = 0  // absolute index where next chunk (center) begins

    // Two-tier transcription state (like Apple's Speech API)
    public private(set) var volatileTokenCount: Int = 0
    public private(set) var confirmedTokenCount: Int = 0

    /// The audio source this stream is configured for
    public var source: AudioSource {
        return audioSource
    }

    // Metrics
    private var startTime: Date?
    private var processedChunks: Int = 0

    /// Initialize the streaming ASR manager
    /// - Parameter config: Configuration for streaming behavior
    public init(config: StreamingAsrConfig = .default) {
        self.config = config

        // Create input stream
        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        self.inputSequence = stream
        self.inputBuilder = continuation

        logger.info(
            "Initialized StreamingAsrManager with config: chunk=\(config.chunkSeconds)s left=\(config.leftContextSeconds)s right=\(config.rightContextSeconds)s"
        )
    }

    /// Start the streaming ASR engine
    /// This will download models if needed and begin processing
    /// - Parameter source: The audio source to use (default: microphone)
    public func start(source: AudioSource = .microphone) async throws {
        logger.info("Starting streaming ASR engine for source: \(String(describing: source))...")

        // Initialize ASR models
        let models = try await AsrModels.downloadAndLoad()
        try await start(models: models, source: source)
    }

    /// Start the streaming ASR engine with pre-loaded models
    /// - Parameters:
    ///   - models: Pre-loaded ASR models to use
    ///   - source: The audio source to use (default: microphone)
    public func start(models: AsrModels, source: AudioSource = .microphone) async throws {
        logger.info(
            "Starting streaming ASR engine with pre-loaded models for source: \(String(describing: source))..."
        )

        self.audioSource = source

        // Initialize ASR manager with provided models
        asrManager = AsrManager(config: config.asrConfig)
        try await asrManager?.initialize(models: models)

        // Reset decoder state for the specific source
        try await asrManager?.resetDecoderState(for: source)

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        accumulatedTokenTimings.removeAll()

        startTime = Date()

        // Start background recognition task
        recognizerTask = Task {
            logger.info("Recognition task started, waiting for audio...")

            for await pcmBuffer in self.inputSequence {
                do {
                    // Convert to 16kHz mono (streaming)
                    let samples = try audioConverter.resampleBuffer(pcmBuffer)

                    // Append to raw sample buffer and attempt windowed processing
                    await self.appendSamplesAndProcess(samples)
                } catch {
                    let streamingError = StreamingAsrError.audioBufferProcessingFailed(error)
                    logger.error(
                        "Audio buffer processing error: \(streamingError.localizedDescription)")
                    await attemptErrorRecovery(error: streamingError)
                }
            }

            // Stream ended: no need to flush converter since each conversion is stateless

            // Then flush remaining assembled audio (no right-context requirement)
            await self.flushRemaining()

            logger.info("Recognition task completed")
        }

        logger.info("Streaming ASR engine started successfully")
    }

    /// Stream audio data for transcription
    /// - Parameter buffer: Audio buffer in any format (will be converted to 16kHz mono)
    public func streamAudio(_ buffer: AVAudioPCMBuffer) {
        inputBuilder.yield(buffer)
    }

    /// Get an async stream of transcription updates
    public var transcriptionUpdates: AsyncStream<StreamingTranscriptionUpdate> {
        AsyncStream { continuation in
            self.updateContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearUpdateContinuation()
                }
            }
        }
    }

    /// Finish streaming and get the final transcription with token timings
    /// - Returns: ASRResult containing the complete transcription text and token timings
    public func finish() async throws -> ASRResult {
        logger.info("Finishing streaming ASR...")

        // Signal end of input
        inputBuilder.finish()

        // Wait for recognition task to complete
        do {
            try await recognizerTask?.value
        } catch {
            logger.error("Recognition task failed: \(error)")
            throw error
        }

        // Convert final accumulated token timings to text
        let finalResult: ASRResult
        if let asrManager = asrManager, !accumulatedTokenTimings.isEmpty {
            // Extract token IDs and timestamps from accumulated TokenTiming instances
            let tokenIds = accumulatedTokenTimings.map { $0.tokenId }
            let timestamps = accumulatedTokenTimings.map { Int($0.startTime / 0.08) }  // Convert to frame timestamps
            let confidences = accumulatedTokenTimings.map { $0.confidence }
            
            let finalAsrResult = asrManager.processTranscriptionResult(
                tokenIds: tokenIds,
                timestamps: timestamps,
                confidences: confidences,
                encoderSequenceLength: 0,
                audioSamples: [],  // Not needed for final text conversion
                processingTime: 0,
                tokenTimings: accumulatedTokenTimings  // Pass the accumulated TokenTiming instances directly
            )
            finalResult = finalAsrResult
        } else {
            finalResult = ASRResult(
                text: "",
                confidence: 0.0,
                duration: Date().timeIntervalSince(startTime ?? Date()),
                processingTime: 0,
                tokenTimings: nil
            )
        }

        logger.info("Final transcription: \(finalResult.text.count) characters")
        return finalResult
    }

    /// Reset the transcriber for a new session
    public func reset() async throws {
        volatileTokenCount = 0
        confirmedTokenCount = 0
        processedChunks = 0
        startTime = Date()
        sampleBuffer.removeAll(keepingCapacity: false)
        bufferStartIndex = 0
        nextWindowCenterStart = 0

        // Reset decoder state for the current audio source
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
        }

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        accumulatedTokenTimings.removeAll()

        logger.info("StreamingAsrManager reset for source: \(String(describing: self.audioSource))")
    }

    /// Cancel streaming without getting results
    public func cancel() async {
        inputBuilder.finish()
        recognizerTask?.cancel()
        updateContinuation?.finish()

        logger.info("StreamingAsrManager cancelled")
    }

    /// Clear the update continuation
    private func clearUpdateContinuation() {
        updateContinuation = nil
    }

    // MARK: - Private Methods

    /// Append new samples and process as many windows as available
    private func appendSamplesAndProcess(_ samples: [Float]) async {
        // Append samples to buffer
        sampleBuffer.append(contentsOf: samples)

        // Process while we have at least chunk + right ahead of the current center start
        let chunk = config.chunkSamples
        let right = config.rightContextSamples
        let left = config.leftContextSamples
        let sampleRate = config.asrConfig.sampleRate

        var currentAbsEnd = bufferStartIndex + sampleBuffer.count
        while currentAbsEnd >= (nextWindowCenterStart + chunk + right) {
            let leftStartAbs = max(0, nextWindowCenterStart - left)
            let rightEndAbs = nextWindowCenterStart + chunk + right
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = rightEndAbs - bufferStartIndex
            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx {
                break
            }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            let actualLeftSecs = Double(nextWindowCenterStart - leftStartAbs) / Double(sampleRate)
            await processWindow(window, actualLeftSeconds: actualLeftSecs)

            // Advance by chunk size
            nextWindowCenterStart += chunk

            // Trim buffer to keep only what's needed for left context
            let trimToAbs = max(0, nextWindowCenterStart - left)
            let dropCount = max(0, trimToAbs - bufferStartIndex)
            if dropCount > 0 && dropCount <= sampleBuffer.count {
                sampleBuffer.removeFirst(dropCount)
                bufferStartIndex += dropCount
            }

            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }
    }

    /// Flush any remaining audio at end of stream (no right-context requirement)
    private func flushRemaining() async {
        let chunk = config.chunkSamples
        let left = config.leftContextSamples
        let sampleRate = config.asrConfig.sampleRate

        var currentAbsEnd = bufferStartIndex + sampleBuffer.count
        while currentAbsEnd > nextWindowCenterStart {  // process until we exhaust
            // If we have less than a chunk ahead, process the final partial chunk
            let availableAhead = currentAbsEnd - nextWindowCenterStart
            if availableAhead <= 0 { break }
            let effectiveChunk = min(chunk, availableAhead)

            let leftStartAbs = max(0, nextWindowCenterStart - left)
            let rightEndAbs = nextWindowCenterStart + effectiveChunk
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = max(rightEndAbs - bufferStartIndex, startIdx)
            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx { break }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            let actualLeftSecs = Double(nextWindowCenterStart - leftStartAbs) / Double(sampleRate)
            await processWindow(window, actualLeftSeconds: actualLeftSecs)

            nextWindowCenterStart += effectiveChunk

            // Trim
            let trimToAbs = max(0, nextWindowCenterStart - left)
            let dropCount = max(0, trimToAbs - bufferStartIndex)
            if dropCount > 0 && dropCount <= sampleBuffer.count {
                sampleBuffer.removeFirst(dropCount)
                bufferStartIndex += dropCount
            }

            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }
    }

    /// Process a single assembled window: [left, chunk, right]
    private func processWindow(_ windowSamples: [Float], actualLeftSeconds: Double) async {
        guard let asrManager = asrManager else { return }

        do {
            let chunkStartTime = Date()

            // Calculate absolute time offset of window start from stream beginning
            let leftStartAbs = max(0, nextWindowCenterStart - config.leftContextSamples)
            let windowTimeOffset = TimeInterval(leftStartAbs) / 16000.0

            // Start frame offset is now handled by decoder's timeJump mechanism

            // Call AsrManager without previous tokens - we'll handle deduplication using timestamps
            let (tokens, timestamps, confidences, _) = try await asrManager.transcribeStreamingChunk(
                windowSamples,
                source: audioSource,
                previousTokens: []  // Empty array - no token-based deduplication
            )

            // Update state
            lastProcessedFrame = max(lastProcessedFrame, timestamps.max() ?? 0)
            segmentIndex += 1

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1

            // Convert window time offset to frame units and add to timestamps
            // This makes timestamps absolute from the beginning of the audio stream
            let windowFrameOffset = Int(windowTimeOffset / 0.08)  // Convert seconds to frames
            let absoluteTimestamps = timestamps.map { $0 + windowFrameOffset }

//            print(">>>BEFORE:", Array(accumulatedTokenTimings.suffix(10)), "\n=====\n",
//                  tokens.map { asrManager.vocabulary[$0]! },
//                  absoluteTimestamps.map { Double($0) * 0.08 },
//                  timestamps.map { Double($0) * 0.08 },
//                  windowTimeOffset, windowTimeOffset + (Double(config.leftContextSamples) / 16000) / 2)
            
            // Deduplicate tokens based on precise timing separation
            let (dedupedTokens, dedupedTimestamps, dedupedConfidences, removedFromAccumulated) = deduplicateTokensByTimestamp(
                tokens: tokens,
                timestamps: absoluteTimestamps,
                confidences: confidences,
                accumulatedTokenTimings: accumulatedTokenTimings,
                currentChunkStartTime: windowTimeOffset
            )
            
            precondition(volatileTokenCount >= removedFromAccumulated, "Number of volatile tokens " +
                         "(\(volatileTokenCount) is less than number of removed tokens (\(removedFromAccumulated))")

            // Remove potentially incorrect tokens from the end of accumulated tokens
//            let removedText = accumulatedTokenTimings[
//                accumulatedTokenTimings.count - removedFromAccumulated..<accumulatedTokenTimings.count
//            ].map { $0.token }.joined()
            if removedFromAccumulated > 0 {
                accumulatedTokenTimings.removeLast(removedFromAccumulated)
            }

//            print(">>>AFTER:", Array(accumulatedTokenTimings.suffix(10)), "\n===== \(removedFromAccumulated)\n",
//                  dedupedTokens.map { asrManager.vocabulary[$0]! }, dedupedTimestamps.map { Double($0) * 0.08 })

            // Convert only the current chunk tokens to text for clean incremental updates
            // The final result will use all accumulated tokens for proper deduplication
            let interim = asrManager.processTranscriptionResult(
                tokenIds: dedupedTokens,  // Use deduplicated tokens
                timestamps: dedupedTimestamps,  // Use deduplicated absolute frame timestamps
                confidences: dedupedConfidences,
                encoderSequenceLength: 0,
                audioSamples: windowSamples,
                processingTime: processingTime
            )

            let tailTimestamp = Int((windowTimeOffset +
                                     (windowTimeOffset > 0 ? config.leftContextSeconds : 0) +
                                     config.chunkSeconds) / 0.08)
//            print("TAIL TIMESTAMP: \(Double(tailTimestamp) * 0.08)")
            var tailTokenCount = 0
            let tokenTimings = interim.tokenTimings!
            for i in (0..<tokenTimings.count).reversed() {
                let tokenTiming = tokenTimings[i]
                tailTokenCount += 1
                if Int(tokenTiming.startTime / 0.08) < tailTimestamp && tokenTiming.token.starts(with: " ") {
                    // Stop on word boundary.
                    break
                }
            }
            
            // Accumulate the token timings from the result
            if let tokenTimings = interim.tokenTimings {
                accumulatedTokenTimings.append(contentsOf: tokenTimings)
            }

            logger.debug(
                "Chunk \(self.processedChunks): '\(interim.text)', time: \(String(format: "%.3f", processingTime))s)"
            )

//            print("TAIL TOKEN COUNT: \(tailTokenCount), REMOVED TOKEN COUNT: \(removedFromAccumulated)")
            
            // Apply confidence-based confirmation logic (uses configured threshold)
            let (curConfirmedTokenCount, curVolatileTokenCount) = await updateTranscriptionState(
                with: interim, tailTokenCount: tailTokenCount)

            if curConfirmedTokenCount != 0 {
                let from = accumulatedTokenTimings.count - (curConfirmedTokenCount + curVolatileTokenCount)
                let to = accumulatedTokenTimings.count - curVolatileTokenCount
                let confirmedText = accumulatedTokenTimingsToText(from: from, to: to)
                let tokenTimings = Array(accumulatedTokenTimings[from..<to])
                
//                print("CONFIRMED TEXT: all: \(accumulatedTokenTimings.count) from: \(from) " +
//                      "to: \(to) removed: \(removedFromAccumulated) tail: \(tailTokenCount) \(confirmedText)")
                
                let update = StreamingTranscriptionUpdate(
                    text: confirmedText,
                    isConfirmed: true,
                    confidence: interim.confidence,
                    timestamp: Date(),
                    tokenTimings: tokenTimings
                )

                updateContinuation?.yield(update)
            }
            
            if curVolatileTokenCount != 0 {
                let from = accumulatedTokenTimings.count - curVolatileTokenCount
                let to = accumulatedTokenTimings.count
                let volatileText = accumulatedTokenTimingsToText(from: from, to: to)
                let tokenTimings = Array(accumulatedTokenTimings[from..<to])

//                print("VOLAITLE TEXT: \(volatileText)")
                
                let update = StreamingTranscriptionUpdate(
                    text: volatileText,
                    isConfirmed: false,
                    confidence: interim.confidence,
                    timestamp: Date(),
                    tokenTimings: tokenTimings
                )

                updateContinuation?.yield(update)
            }
            
            // Emit update based on progressive confidence model
//            let totalAudioProcessed = Double(bufferStartIndex + sampleBuffer.count) / 16000.0
//            let hasMinimumContext = totalAudioProcessed >= config.minContextForConfirmation
//            let isHighConfidence = Double(interim.confidence) >= config.confirmationThreshold
//            let shouldConfirm = isHighConfidence && hasMinimumContext
//
//            let update = StreamingTranscriptionUpdate(
//                text: interim.text,
//                removedText: removedText,
//                previousText: previousText,
//                isConfirmed: shouldConfirm,
//                confidence: interim.confidence,
//                timestamp: Date(),
//                tokenTimings: interim.tokenTimings
//            )
//
//            updateContinuation?.yield(update)

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }
    
    /// Converts a specified subset of accumulated token timings to text.
    private func accumulatedTokenTimingsToText(from: Int, to: Int) -> String {
        return accumulatedTokenTimings[from..<to]
            .map { $0.token }
            .joined()
            .replacingOccurrences(of: "▁", with: " ")
    }

    /// Update transcription state based on confidence and context duration
    private func updateTranscriptionState(with result: ASRResult,
                                          tailTokenCount: Int) async -> (Int, Int) {
        let totalAudioProcessed = Double(bufferStartIndex + sampleBuffer.count) / 16000.0
        let hasMinimumContext = totalAudioProcessed >= config.minContextForConfirmation
        let isHighConfidence = Double(result.confidence) >= config.confirmationThreshold

        // Progressive confidence model:
        // 1. Always show text immediately as volatile for responsiveness
        // 2. Only confirm text when we have both high confidence AND sufficient context
        let shouldConfirm = isHighConfidence && hasMinimumContext
        
        if shouldConfirm {
            let newConfirmedTokenCount = accumulatedTokenTimings.count - tailTokenCount
            let curConfirmedTokenCount = newConfirmedTokenCount - confirmedTokenCount
            confirmedTokenCount = newConfirmedTokenCount
            volatileTokenCount = tailTokenCount
            
            logger.debug(
                "CONFIRMED (\(result.confidence), \(String(format: "%.1f", totalAudioProcessed))s context): promoted to confirmed; new volatile '\(result.text)'"
            )
            return (curConfirmedTokenCount, tailTokenCount)
        } else {
            // Only update volatile text (hypothesis)
            volatileTokenCount += result.tokenTimings!.count
            let reason =
                !hasMinimumContext
                ? "insufficient context (\(String(format: "%.1f", totalAudioProcessed))s)" : "low confidence"
            logger.debug("VOLATILE (\(result.confidence)): \(reason) - updated volatile '\(result.text)'")
            return (0, result.tokenTimings!.count)
        }
    }

    /// Attempt to recover from processing errors
    private func attemptErrorRecovery(error: Error) async {
        logger.warning("Attempting error recovery for: \(error)")

        // Handle specific error types with targeted recovery
        if let streamingError = error as? StreamingAsrError {
            switch streamingError {
            case .modelsNotLoaded:
                logger.error("Models not loaded - cannot recover automatically")

            case .streamAlreadyExists:
                logger.error("Stream already exists - cannot recover automatically")

            case .audioBufferProcessingFailed:
                logger.info("Recovering from audio buffer error")

            case .audioConversionFailed:
                logger.info("Recovering from audio conversion error")

            case .modelProcessingFailed:
                logger.info("Recovering from model processing error - resetting decoder state")
                await resetDecoderForRecovery()

            case .bufferOverflow:
                logger.info("Buffer overflow handled automatically")

            case .invalidConfiguration:
                logger.error("Configuration error cannot be recovered automatically")
            }
        } else {
            // Generic recovery for non-streaming errors
            await resetDecoderForRecovery()
        }
    }

    /// Reset decoder state for error recovery
    private func resetDecoderForRecovery() async {
        if let asrManager = asrManager {
            do {
                try await asrManager.resetDecoderState(for: audioSource)
                logger.info("Successfully reset decoder state during error recovery")
            } catch {
                logger.error("Failed to reset decoder state during recovery: \(error)")

                // Last resort: try to reinitialize the ASR manager
                do {
                    let models = try await AsrModels.downloadAndLoad()
                    let newAsrManager = AsrManager(config: config.asrConfig)
                    try await newAsrManager.initialize(models: models)
                    self.asrManager = newAsrManager
                    logger.info("Successfully reinitialized ASR manager during error recovery")
                } catch {
                    logger.error("Failed to reinitialize ASR manager during recovery: \(error)")
                }
            }
        }
    }
    
    /// Deduplicate tokens based on precise timing separation
    /// Uses the exact overlap between chunks to determine the separation point
    private func deduplicateTokensByTimestamp(
        tokens: [Int],
        timestamps: [Int],
        confidences: [Float],
        accumulatedTokenTimings: [TokenTiming],
        currentChunkStartTime: TimeInterval
    ) -> (tokens: [Int], timestamps: [Int], confidences: [Float], removedFromAccumulated: Int) {
        
        guard !tokens.isEmpty && !accumulatedTokenTimings.isEmpty else {
            return (tokens, timestamps, confidences, 0)
        }
        
        let punctuationTokens = [7883, 7952, 7948, 7877, 7956, 8020]
//        let delta: Double = 0.08 / 2 + 0.01 // Possible token misplacement.
        
        // Calculate the separation time based on the overlap determined by leftContextSamples
        // Current chunk start time in frame units
//        let currentChunkStartFrame = Int(currentChunkStartTime / 0.08)
        
        // The overlap is determined by leftContextSamples - this is how much the current chunk
        // overlaps with the previous chunk
//        let leftContextFrames = Int(Double(config.leftContextSamples) / 16000.0 / 0.08)  // Convert samples to frames
        
        // Calculate separation point: current chunk start + half of the overlap
//        let separationFrame = currentChunkStartFrame + (leftContextFrames / 2)
//        let separationFrame = Int((accumulatedTokenTimings.last!.endTime + Double(timestamps[0]) * 0.08) * 0.5 / 0.08)
//        print("SEPARATION TIME", Double(separationTime) * 0.08)
        
        // Skip all leading punctuation tokens.
//        var firstTimestamp: Int = 0
//        for i in 0..<tokens.count {
//            if !punctuationTokens.contains(tokens[i]) {
//                firstTimestamp = timestamps[i]
//                break
//            }
//        }
//        
//        guard firstTimestamp != 0 else {
//            return (tokens, timestamps, confidences, 0)
//        }
//        
//        let separationTime = (accumulatedTokenTimings.last!.endTime + Double(firstTimestamp) * 0.08) * 0.5

        var separationTime = currentChunkStartTime + config.leftContextSeconds / 2
//        var earlySeparationTime = separationTime
        
        // We need to keep tokens which only found in the new chunk, but missing from previous.
        let lastAccumulatedTokenEndTime = accumulatedTokenTimings.last!.endTime
        separationTime = min(
            separationTime + config.leftContextSeconds / 2 + config.rightContextSeconds / 2,
            lastAccumulatedTokenEndTime)
//        if separationTime <= lastAccumulatedTokenEndTime {
//            separationTime += config.leftContextSeconds / 2
////            earlySeparationTime = separationTime
//            if separationTime <= lastAccumulatedTokenEndTime {
//                separationTime += config.rightContextSeconds / 2
//            }
//        }

        let separationFrame = Int(round(separationTime / 0.08))
//        let earlySeparationFrame = Int(round(earlySeparationTime / 0.08))
        
//        print("<><><>SEPARATION TIME", Double(separationTime), separationFrame, currentChunkStartTime)

        // Keep tokens from new chunk that are at or after the separation point
        
        var firstValidTokenIndex: Int = tokens.count
        for i in 0..<tokens.count {
            // Skip first token if it's punctuation.
            if i == 0 && punctuationTokens.contains(tokens[i]) {
                continue
            }
            
            let tokenTimestamp = timestamps[i]
//            let token = asrManager!.vocabulary[tokens[i]]!
            if tokenTimestamp >= separationFrame {
//            if tokenTimestamp >= separationFrame || (tokenTimestamp >= earlySeparationFrame &&
//                                                     startsWithWhitespaceAndUppercase(token)) {
                // Stop on first token which is on or after separation time, or if a capital letter
                // is encountered (assuming the model is confident enough here).
                firstValidTokenIndex = i
                break
//            let tokenTimestamp = Double(timestamps[i]) * 0.08
//            if tokenTimestamp > separationTime {
                
//                keptTokens.append(tokens[i])
//                keptTimestamps.append(timestamps[i])
//                keptConfidences.append(confidences[i])
            }
        }
        
        var keptTokens: [Int] = Array(tokens[firstValidTokenIndex...])
        var keptTimestamps: [Int] = Array(timestamps[firstValidTokenIndex...])
        var keptConfidences: [Float] = Array(confidences[firstValidTokenIndex...])
        
        guard !keptTokens.isEmpty else {
            return (keptTokens, keptTimestamps, keptConfidences, 0)
        }

//        // Move separation time to the beginnig of the first kept token.
//        separationTime = Double(keptTimestamps[0]) * 0.08
//        separationFrame = Int(round(separationTime / 0.08))

        let firstKeptTokenTime = Double(keptTimestamps[0]) * 0.08
        let firstKeptTokenFrame = Int(round(firstKeptTokenTime / 0.08))
        
//        print("SEPARATION TIME (2)", Double(separationTime), separationFrame)

        // Remove tokens from accumulated that are after the separation point
        // Keep tokens that are exactly at the separation point
        var removedFromAccumulated = 0
        for i in (0..<accumulatedTokenTimings.count).reversed() {
            let tokenStartFrame = Int(round(accumulatedTokenTimings[i].startTime / 0.08))
//            if i > 0 {
//                // Remove the token if previous one has the same start time. Bug?
//                let prvTokenStartFrame = Int(round(accumulatedTokenTimings[i - 1].startTime / 0.08))
//                if prvTokenStartFrame == tokenStartFrame {
//                    removedFromAccumulated += 1
//                    continue
//                }
//            }
            if tokenStartFrame > firstKeptTokenFrame {
                removedFromAccumulated += 1
                continue
            } else if punctuationTokens.contains(accumulatedTokenTimings[i].tokenId) {
                // If punctuation is found just before the kept tokens, stop removing accumulated tokens.
                break
            }
            if tokenStartFrame > separationFrame {
//            let tokenStartTime = accumulatedTokenTimings[i].startTime
//            if tokenStartTime > separationTime {
                removedFromAccumulated += 1
            } else {
                break
            }
        }
        
//        print("REMOVED FROM ACCUMULATED: \(removedFromAccumulated)")
        
        // Handle boundary case: if we have tokens at the separation point from both sides,
        // compare the last remaining accumulated token with the first remaining new token
        if removedFromAccumulated < accumulatedTokenTimings.count && !keptTokens.isEmpty {
            let lastAccumulatedIndex = accumulatedTokenTimings.count - 1 - removedFromAccumulated
            let lastAccumulatedTokenTiming = accumulatedTokenTimings[lastAccumulatedIndex]

            if punctuationTokens.contains(lastAccumulatedTokenTiming.tokenId) && punctuationTokens.contains(keptTokens[0]) {
                // Punctuation tokens on both ends. Keep the new one.
                removedFromAccumulated += 1
            } else {
                let lastAccumulatedTokenStartFrame = Int(round(lastAccumulatedTokenTiming.startTime / 0.08))
                let lastAccumulatedTokenEndFrame = Int(round(lastAccumulatedTokenTiming.endTime / 0.08))
                let firstNewTokenTimestamp = keptTimestamps[0]
                let firstKeptToken = asrManager!.vocabulary[keptTokens[0]]!

                if firstNewTokenTimestamp >= lastAccumulatedTokenStartFrame &&
                    firstNewTokenTimestamp <= lastAccumulatedTokenEndFrame {
                    // Last accumulated token and first new token intersects.
//                    print("INTERSECTING TOKENS", lastAccumulatedTokenTiming.token, firstKeptToken)
                    if lastAccumulatedTokenTiming.tokenId == keptTokens[0] {
                        // Same token at boundary - remove the duplicate from new tokens
                        keptTokens.removeFirst()
                        keptTimestamps.removeFirst()
                        keptConfidences.removeFirst()
                    } else if firstKeptToken.lowercased() == lastAccumulatedTokenTiming.token.lowercased() {
                        // We have the same token at the boundary, but in different case. Remove
                        // duplicate from new tokens.
                        keptTokens.removeFirst()
                        keptTimestamps.removeFirst()
                        keptConfidences.removeFirst()
                    } else if startsWithWhitespaceAndUppercase(lastAccumulatedTokenTiming.token) &&
                                startsWithWhitespaceAndUppercase(firstKeptToken) {
                        // We have two "space + capital letter" at the boundary. Most likely it's a
                        // misinterpreted beginning of a sentence. Keep new one.
                        // FIXME: It may cause small issues in the middle of a sentence with proper names.
                        removedFromAccumulated += 1
                    } else if keptTokens.count > 1 && removedFromAccumulated < accumulatedTokenTimings.count - 1 {
                        // We have at least two tokens on both side.
                        let preLastAccumulatedTokenTiming = accumulatedTokenTimings[lastAccumulatedIndex - 1]
                        let secondKeptToken = asrManager!.vocabulary[keptTokens[1]]!
//                        if keptTokens[1] == lastAccumulatedTokenTiming.tokenId &&
//                            keptTokens[0] == preLastAccumulatedTokenTiming.tokenId {
                        if secondKeptToken.lowercased() == lastAccumulatedTokenTiming.token.lowercased() &&
                            firstKeptToken.lowercased() == preLastAccumulatedTokenTiming.token.lowercased() {
                            // Two tokens are the same at the boundary. Remove them both.
                            // TODO: Check timings if misfire.
                            keptTokens.removeFirst(2)
                            keptTimestamps.removeFirst(2)
                            keptConfidences.removeFirst(2)
                        } else if lastAccumulatedTokenTiming.token.lowercased() ==
                                    (firstKeptToken + secondKeptToken).lowercased() {
                            // Last accumulated token is the same as the two first accumulated tokens. Remove the new tokens.
                            keptTokens.removeFirst(2)
                            keptTimestamps.removeFirst(2)
                            keptConfidences.removeFirst(2)
                        } else if (preLastAccumulatedTokenTiming.token + lastAccumulatedTokenTiming.token).lowercased() ==
                                    firstKeptToken.lowercased() {
                            // Two last accumulated tokens are the same as the first new one. Remove the new one.
                            keptTokens.removeFirst(1)
                            keptTimestamps.removeFirst(1)
                            keptConfidences.removeFirst(1)
                        } else if firstNewTokenTimestamp == Int(round(preLastAccumulatedTokenTiming.endTime / 0.08)) &&
                                    keptTokens[0] == preLastAccumulatedTokenTiming.tokenId {
                            // Most likely we have a mistakenly recognized token at the end of accumulated
                            // tokens. Thus remove two previously accumulated tokens.
                            removedFromAccumulated += 2
                            //                        print("REMOVED FROM ACCUMULATED (2): \(removedFromAccumulated)")
                        }
                    }
                }
            }
            
//            let firstNewTokenStartTime = Double(keptTimestamps[0]) * 0.08
//            
//            if lastAccumulatedTokenTiming.startTime <= firstNewTokenStartTime &&
//                lastAccumulatedTokenTiming.endTime >= firstNewTokenStartTime {
//                let firstNewToken = keptTokens[0]
//                
//                if lastAccumulatedTokenTiming.tokenId == firstNewToken {
//                    // Same token at boundary - remove the duplicate from new tokens
//                    keptTokens.removeFirst()
//                    keptTimestamps.removeFirst()
//                    keptConfidences.removeFirst()
//                }
//            }
            
//            let lastAccumulatedTokenStartFrame = Int(accumulatedTokenTimings[lastAccumulatedIndex].startTime / 0.08)
//            let firstNewTokenTimestamp = keptTimestamps[0]
            
            // Only compare if both tokens are actually at the separation boundary
//            if lastAccumulatedTokenStartFrame == separationFrame && firstNewTokenTimestamp == separationFrame {
//                let lastAccumulatedToken = accumulatedTokenTimings[lastAccumulatedIndex].tokenId
//                let firstNewToken = keptTokens[0]
//                
//                if lastAccumulatedToken == firstNewToken {
//                    // Same token at boundary - remove the duplicate from new tokens
//                    keptTokens.removeFirst()
//                    keptTimestamps.removeFirst()
//                    keptConfidences.removeFirst()
//                }
                // If different tokens, keep both (they represent different content)
//            }
        }
                
        return (keptTokens, keptTimestamps, keptConfidences, removedFromAccumulated)
    }
    
    // Checks if given string contains whitespace as the first character and an uppercase letter as
    // the second.
    func startsWithWhitespaceAndUppercase(_ s: String) -> Bool {
        guard s.count >= 2 else {
            return false
        }
        return s.first!.isWhitespace && s[s.index(s.startIndex, offsetBy: 1)].isUppercase
    }
}

/// Configuration for StreamingAsrManager
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingAsrConfig: Sendable {
    /// Main chunk size for stable transcription (seconds). Should be 10-11s for best quality
    public let chunkSeconds: TimeInterval
    /// Quick hypothesis chunk size for immediate feedback (seconds). Typical: 1.0s
    public let hypothesisChunkSeconds: TimeInterval
    /// Left context appended to each window (seconds). Typical: 10.0s
    public let leftContextSeconds: TimeInterval
    /// Right context lookahead (seconds). Typical: 2.0s (adds latency)
    public let rightContextSeconds: TimeInterval
    /// Minimum audio duration before confirming text (seconds). Should be ~10s
    public let minContextForConfirmation: TimeInterval

    /// Confidence threshold for promoting volatile text to confirmed (0.0...1.0)
    public let confirmationThreshold: Double

    /// Default configuration aligned with previous API expectations
    public static let `default` = StreamingAsrConfig(
        chunkSeconds: 15.0,
        hypothesisChunkSeconds: 2.0,
        leftContextSeconds: 10.0,
        rightContextSeconds: 2.0,
        minContextForConfirmation: 10.0,
        confirmationThreshold: 0.85
    )

    /// Optimized streaming configuration: Dual-track processing for best experience
    /// Uses ChunkProcessor's proven 11-2-2 approach for stable transcription
    /// Plus quick hypothesis updates for immediate feedback
    public static let streaming = StreamingAsrConfig(
        chunkSeconds: 11.0,  // Match ChunkProcessor for stable transcription
        hypothesisChunkSeconds: 1.0,  // Quick hypothesis updates
        leftContextSeconds: 2.0,  // Match ChunkProcessor left context
        rightContextSeconds: 2.0,  // Match ChunkProcessor right context
        minContextForConfirmation: 10.0,  // Need sufficient context before confirming
        confirmationThreshold: 0.80  // Higher threshold for more stable confirmations
    )

    public init(
        chunkSeconds: TimeInterval = 10.0,
        hypothesisChunkSeconds: TimeInterval = 1.0,
        leftContextSeconds: TimeInterval = 2.0,
        rightContextSeconds: TimeInterval = 2.0,
        minContextForConfirmation: TimeInterval = 10.0,
        confirmationThreshold: Double = 0.85
    ) {
        self.chunkSeconds = chunkSeconds
        self.hypothesisChunkSeconds = hypothesisChunkSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.minContextForConfirmation = minContextForConfirmation
        self.confirmationThreshold = confirmationThreshold
    }

    /// Backward-compatible convenience initializer used by tests (chunkDuration label)
    public init(
        confirmationThreshold: Double = 0.85,
        chunkDuration: TimeInterval
    ) {
        self.init(
            chunkSeconds: chunkDuration,
            hypothesisChunkSeconds: min(1.0, chunkDuration / 2.0),  // Default to half chunk duration
            leftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            minContextForConfirmation: 10.0,
            confirmationThreshold: confirmationThreshold
        )
    }

    /// Custom configuration factory expected by tests
    public static func custom(
        chunkDuration: TimeInterval,
        confirmationThreshold: Double
    ) -> StreamingAsrConfig {
        StreamingAsrConfig(
            chunkSeconds: chunkDuration,
            hypothesisChunkSeconds: min(1.0, chunkDuration / 2.0),  // Default to half chunk duration
            leftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            minContextForConfirmation: 10.0,
            confirmationThreshold: confirmationThreshold
        )
    }

    // Internal ASR configuration
    var asrConfig: ASRConfig {
        ASRConfig(
            sampleRate: 16000,
            tdtConfig: TdtConfig()
        )
    }

    // Sample counts at 16 kHz
    var chunkSamples: Int { Int(chunkSeconds * 16000) }
    var hypothesisChunkSamples: Int { Int(hypothesisChunkSeconds * 16000) }
    var leftContextSamples: Int { Int(leftContextSeconds * 16000) }
    var rightContextSamples: Int { Int(rightContextSeconds * 16000) }
    var minContextForConfirmationSamples: Int { Int(minContextForConfirmation * 16000) }

    // Backward-compat convenience for existing call-sites/tests
    var chunkDuration: TimeInterval { chunkSeconds }
    var bufferCapacity: Int { Int(15.0 * 16000) }
    var chunkSizeInSamples: Int { chunkSamples }
}

/// Transcription update from streaming ASR
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptionUpdate: Sendable {
    /// The transcribed text (update).
    public let text: String
    
    /// Whether this text is confirmed (high confidence) or volatile (may change)
    public let isConfirmed: Bool

    /// Confidence score (0.0 - 1.0)
    public let confidence: Float

    /// Timestamp of this update
    public let timestamp: Date

    /// Token timings
    public let tokenTimings: [TokenTiming]?

    public init(
        text: String,
        isConfirmed: Bool,
        confidence: Float,
        timestamp: Date,
        tokenTimings: [TokenTiming]?
    ) {
        self.text = text
        self.isConfirmed = isConfirmed
        self.confidence = confidence
        self.timestamp = timestamp
        self.tokenTimings = tokenTimings
    }
}
