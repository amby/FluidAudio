import Accelerate
import Foundation
import OSLog

/// In-memory speaker database for streaming diarization
/// Tracks speakers across chunks and maintains consistent IDs
@available(macOS 13.0, iOS 16.0, *)
public class SpeakerManager {
    internal let logger = AppLogger(category: "SpeakerManager")

    // Constants
    public static let embeddingSize = 256  // Standard embedding dimension for speaker models

    // Speaker database: ID -> Speaker
    internal var speakerDatabase: [String: Speaker] = [:]
    // All currently clusterized embeddings.
    internal var embeddings: [[Float]] = []
    // Computed minimal cluster distances.
    internal var minClusterDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
    private var nextSpeakerId = 1
    internal let queue = DispatchQueue(label: "speaker.manager.queue", attributes: .concurrent)

    // Track the highest speaker ID to ensure uniqueness
    private var highestSpeakerId = 0

    public var speakerThreshold: Float  // Max distance for speaker assignment (default: 0.65)
    public var embeddingThreshold: Float  // Max distance for updating embeddings (default: 0.45)
    public var minSpeechDuration: Float  // Min duration to create speaker (default: 1.0)
    public var minEmbeddingUpdateDuration: Float  // Min duration to update embeddings (default: 2.0)
    private var maxEmbeddingsPerSpeaker: Int = 50 // Maximum number of embeddings to store per speaker.

    public init(
        speakerThreshold: Float = 0.65,
        embeddingThreshold: Float = 0.45,
        minSpeechDuration: Float = 1.0,
        minEmbeddingUpdateDuration: Float = 2.0,
        maxEmbeddingsPerSpeaker: Int = 50
    ) {
        self.speakerThreshold = speakerThreshold
        self.embeddingThreshold = embeddingThreshold
        self.minSpeechDuration = minSpeechDuration
        self.minEmbeddingUpdateDuration = minEmbeddingUpdateDuration
        self.maxEmbeddingsPerSpeaker = maxEmbeddingsPerSpeaker
    }

    public func initializeKnownSpeakers(_ speakers: [Speaker]) {
        queue.sync(flags: .barrier) {
            var maxNumericId = 0

            for speaker in speakers {
                guard speaker.currentEmbedding.count == Self.embeddingSize else {
                    logger.warning(
                        "Skipping speaker \(speaker.id) - invalid embedding size: \(speaker.currentEmbedding.count)")
                    continue
                }

                speaker.clusterized = false
                speakerDatabase[speaker.id] = speaker
                embeddings.append(speaker.currentEmbedding)

                // Try to extract numeric ID if it's a pure number
                if let numericId = Int(speaker.id) {
                    maxNumericId = max(maxNumericId, numericId)
                }

                logger.info(
                    "Initialized known speaker: \(speaker.id) with \(speaker.rawEmbeddings.count) raw embeddings"
                )
            }

            self.highestSpeakerId = maxNumericId
            self.nextSpeakerId = maxNumericId + 1

            logger.info(
                "Initialized with \(self.speakerDatabase.count) known speakers, next ID will be: \(self.nextSpeakerId)"
            )
        }
    }
    
    // Assigns speakers to given embeddings. Also returns absolute indices for valid embeddings and
    // -1 for all other embeddings.
    public func assignSpeakers(embeddings newEmbeddings: [[Float]],
                               durations: [Float],
                               confidences: [Float]) -> ([Speaker?], [Int]) {
        precondition(newEmbeddings.count == durations.count && durations.count == confidences.count,
                     "Mismatched number of embeddings (\(newEmbeddings.count)), " +
                     "durations (\(durations.count)) and confidences (\(confidences.count))")
        
        guard !newEmbeddings.isEmpty else {
            return ([], [])
        }
        
        var newEmbeddingsToSpeakersIDs: [Int: String] = [:]
        var newEmbeddingIndices: [Int] = []
        return queue.sync(flags: .barrier) {
            var validEmbeddingIndices: [Int] = []
            var recheckEmbeddingIndices: [Int] = []
            for (i, embedding) in newEmbeddings.enumerated() {
                guard !embedding.isEmpty, embedding.count == Self.embeddingSize else {
                    newEmbeddingIndices.append(-1)
                    continue
                }
                guard durations[i] >= minSpeechDuration else {
                    recheckEmbeddingIndices.append(i)
                    newEmbeddingIndices.append(-1)
                    continue
                }
                newEmbeddingIndices.append(embeddings.count)
                embeddings.append(embedding)
                validEmbeddingIndices.append(i)
            }
            
            guard !validEmbeddingIndices.isEmpty || !recheckEmbeddingIndices.isEmpty else {
                return (.init(repeating: nil, count: newEmbeddings.count), newEmbeddingIndices)
            }
            
            if !validEmbeddingIndices.isEmpty {
                // Perform clusterization of all embedding collected so far.
                // FIXME: Minimum cluster count should depend on number of inital speakers. Probably.
                let (minClusterDistances, clusters) = clusterize(
                    maxDistance: speakerThreshold,
                    minClusterCount: speakerDatabase.values.count { $0.clusterized },
                    embeddings: embeddings,
                    minClusterDistances: minClusterDistances)
                
//                print("MIN CLUSTER DISTANCES: \(embeddings.count) \(minClusterDistances)")
//                print("CLUSTERS: \(clusters.map { $0.embeddingIndices } )")
                self.minClusterDistances = minClusterDistances
                
                let validNewEmbeddingsStartIndex = embeddings.count - validEmbeddingIndices.count
                let validNewEmbeddingsToClusters: [Int: Int] = clusters.enumerated().reduce(into: [:]) {
                    for embeddingIndex in $1.element.embeddingIndices {
                        if embeddingIndex >= validNewEmbeddingsStartIndex {
                            $0[embeddingIndex - validNewEmbeddingsStartIndex] = $1.offset
                        }
                    }
                }
                
                // Associate existing speakers to the new clusters.
                let speakers = Array(speakerDatabase.values)
                let speakerEmbeddings: [[Float]] = speakers.reduce(into: []) {
                    $0.append($1.currentEmbedding)
                }
                
                var minDistanceToClusters = computeDistancesToClusters(
                    embeddings: speakerEmbeddings,
                    clusters: clusters,
                    distancesToClusters: ClusterDistances(type: .min, distances: [], otherIndices: []))
                
                var clustersToUsers: [Int: String] = [:]
                var tmpClusters = clusters
                while true {
                    let (uIntMinIndex, minDistance) = vDSP.indexOfMinimum(minDistanceToClusters.distances)
                    // FIXME: For initially known speakers we may need to use a smaller threshold
                    // to avoid matching them with someone else.
                    if minDistance >= speakerThreshold {
                        break
                    }
                    let minIndex = Int(uIntMinIndex)
                    
                    let clusterIndex = minDistanceToClusters.otherIndices[minIndex]
                    let speaker = speakers[minIndex]
                    
                    clustersToUsers[clusterIndex] = speaker.id

                    speaker.currentEmbedding = clusters[clusterIndex].centroid
                    speaker.clusterized = true

                    tmpClusters[clusterIndex] = Cluster(embeddingIndices: [], centroid: [])
                    
                    minDistanceToClusters = computeDistancesToClusters(
                        embeddings: speakerEmbeddings,
                        clusters: tmpClusters,
                        distancesToClusters: minDistanceToClusters,
                        removedEmbeddingIndex: minIndex,
                        removedClusterIndex: clusterIndex)
                }
                
                // Add new users for remaining clusters.
                for (clusterIndex, cluster) in tmpClusters.enumerated() where !cluster.centroid.isEmpty {
                    let speakerID = createNewSpeaker(embedding: cluster.centroid, duration: 0.0)
                    speakerDatabase[speakerID]!.clusterized = true
                    clustersToUsers[clusterIndex] = speakerID
                }
                
//                print("CLUSTERS TO USERS: \(clustersToUsers) \(newEmbeddings.count) \(validEmbeddingIndices) \(recheckEmbeddingIndices)")
                
                // Map valid embeddings to their speaker IDs.
                for (i, validEmbeddingIndex) in validEmbeddingIndices.enumerated() {
                    newEmbeddingsToSpeakersIDs[validEmbeddingIndex] =
                        clustersToUsers[validNewEmbeddingsToClusters[i]!]
                }
                
                // Get rid of excessive embeddings (farthest ones from centroid of each cluster).
                var removedEmbeddingIndices: Set<Int> = []
                for cluster in clusters {
                    guard cluster.embeddingIndices.count > maxEmbeddingsPerSpeaker else {
                        continue
                    }
                    let clusterEmbeddings = cluster.embeddingIndices.reduce(into: [[Float]]()) {
                        $0.append(embeddings[$1])
                    }
                    var maxDistanceToClusters = computeDistancesToClusters(
                        embeddings: clusterEmbeddings,
                        clusters: [cluster],
                        distancesToClusters: ClusterDistances(type: .max, distances: [], otherIndices: []))

                    var removedClusterEmbeddingsCount: Int = 0
                    while cluster.embeddingIndices.count - removedClusterEmbeddingsCount > maxEmbeddingsPerSpeaker {
                        let (uIntMaxIndex, _) = vDSP.indexOfMaximum(maxDistanceToClusters.distances)
                        let maxIndex = Int(uIntMaxIndex)
                        
                        removedEmbeddingIndices.insert(cluster.embeddingIndices[maxIndex])
                        removedClusterEmbeddingsCount += 1

                        maxDistanceToClusters.distances[maxIndex] = -.infinity
                        
//                        maxDistanceToClusters = computeDistancesToClusters(
//                            embeddings: clusterEmbeddings,
//                            clusters: clusters,
//                            distancesToClusters: maxDistanceToClusters,
//                            removedEmbeddingIndex: maxIndex)
                    }
                }

                if !removedEmbeddingIndices.isEmpty {
                    embeddings = embeddings
                        .enumerated()
                        .filter { !removedEmbeddingIndices.contains($0.offset) }
                        .map { $0.element }
                    
                    self.minClusterDistances = ClusterDistances(
                        type: minClusterDistances.type,
                        distances: minClusterDistances.distances
                            .enumerated()
                            .filter { !removedEmbeddingIndices.contains($0.offset) }
                            .map { $0.element },
                        otherIndices: minClusterDistances.otherIndices
                            .enumerated()
                            .filter { !removedEmbeddingIndices.contains($0.offset) }
                            .map { $0.element }
                    )
                }
            }
            
            // Recheck embeddings that was filtered out due to minSpeechDuration. They may be
            // associated with speakers, but should not be used for clustering.
            for i in recheckEmbeddingIndices {
                let (closesetSpeakerID, distance) = findClosestSpeaker(to: newEmbeddings[i])
                guard let closesetSpeakerID, distance < speakerThreshold else {
                    // Corresponding speaker not found. Ignored.
                    continue
                }
                newEmbeddingsToSpeakersIDs[i] = closesetSpeakerID
            }
            
            // Update durations and updatedAt of speakers.
            for (i, speakerID) in newEmbeddingsToSpeakersIDs {
                let speaker = speakerDatabase[speakerID]!
                speaker.duration += durations[i]
                speaker.updatedAt = Date()
            }
            
            return ((0..<newEmbeddings.count).map { speakerDatabase[newEmbeddingsToSpeakersIDs[$0] ?? ""] },
                    newEmbeddingIndices)
        }
    }

    public func assignSpeaker(
        _ embedding: [Float],
        speechDuration: Float,
        confidence: Float = 1.0
    ) -> Speaker? {
        guard !embedding.isEmpty && embedding.count == Self.embeddingSize else {
            logger.error("Invalid embedding size: \(embedding.count)")
            return nil
        }

        return queue.sync(flags: .barrier) {
            let (closestSpeaker, distance) = findClosestSpeaker(to: embedding)

            if let speakerId = closestSpeaker, distance < speakerThreshold {
                updateExistingSpeaker(
                    speakerId: speakerId,
                    embedding: embedding,
                    duration: speechDuration,
                    distance: distance
                )

                if let speaker = speakerDatabase[speakerId] {
                    return speaker
                }
                return nil
            }

            // Step 3: Create new speaker if duration is sufficient
            if speechDuration >= minSpeechDuration {
                let newSpeakerId = createNewSpeaker(
                    embedding: embedding,
                    duration: speechDuration,
//                    distanceToClosest: distance
                )

                // Return the Speaker object
                if let speaker = speakerDatabase[newSpeakerId] {
                    return speaker
                }
                return nil
            }

            // Step 4: Audio segment too short
            logger.debug("Audio segment too short (\(speechDuration)s) to create new speaker")
            return nil
        }
    }

    public func findClosestSpeaker(to embedding: [Float]) -> (speakerId: String?, distance: Float) {
        var minDistance: Float = Float.infinity
        var closestSpeakerId: String?

        for (speakerId, speaker) in speakerDatabase {
            let distance = cosineDist(embedding, speaker.currentEmbedding)
            if distance < minDistance {
                minDistance = distance
                closestSpeakerId = speakerId
            }
        }

        return (closestSpeakerId, minDistance)
    }

    private func updateExistingSpeaker(
        speakerId: String,
        embedding: [Float],
        duration: Float,
        distance: Float
    ) {
        guard let speaker = speakerDatabase[speakerId] else {
            logger.error("Speaker \(speakerId) not found in database")
            return
        }

        // Update embedding if quality is good
        if distance < embeddingThreshold {
            let embeddingMagnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
            if embeddingMagnitude > 0.1 {
                speaker.updateMainEmbedding(
                    duration: duration,
                    embedding: embedding,
                    segmentId: UUID(),
                    alpha: 0.9
                )
            }
        } else {
            // Just update duration if not updating embedding
            speaker.duration += duration
            speaker.updatedAt = Date()
        }

        speakerDatabase[speakerId] = speaker
    }

    private func createNewSpeaker(
        embedding: [Float],
        duration: Float
//        distanceToClosest: Float
    ) -> String {
        let newSpeakerId = String(nextSpeakerId)
        nextSpeakerId += 1
        highestSpeakerId = max(highestSpeakerId, nextSpeakerId - 1)

        // Create new Speaker object
        let newSpeaker = Speaker(
            id: newSpeakerId,
            name: "Speaker \(newSpeakerId)",  // Default name with number
            currentEmbedding: embedding,
            duration: duration
        )

        // Add initial raw embedding
//        let initialRaw = RawEmbedding(segmentId: UUID(), embedding: embedding, timestamp: Date())
//        newSpeaker.addRawEmbedding(initialRaw)

        speakerDatabase[newSpeakerId] = newSpeaker

//        logger.info("Created new speaker \(newSpeakerId) (distance to closest: \(distanceToClosest))")
        logger.info("Created new speaker \(newSpeakerId)")
        return newSpeakerId
    }

    /// Internal cosine distance calculation that delegates to SpeakerUtilities
    /// Kept for backward compatibility with tests
    internal func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        return SpeakerUtilities.cosineDistance(a, b)
    }

    public var speakerCount: Int {
        queue.sync { speakerDatabase.count }
    }

    public var speakerIds: [String] {
        queue.sync { Array(speakerDatabase.keys).sorted() }
    }

    /// Get all speakers (for testing/debugging).
    public func getAllSpeakers() -> [String: Speaker] {
        queue.sync {
            return speakerDatabase
        }
    }

    public func getSpeaker(for speakerId: String) -> Speaker? {
        queue.sync { speakerDatabase[speakerId] }
    }

    /// - Parameter speaker: The Speaker object to upsert
    public func upsertSpeaker(_ speaker: Speaker) {
        upsertSpeaker(
            id: speaker.id,
            currentEmbedding: speaker.currentEmbedding,
            duration: speaker.duration,
            rawEmbeddings: speaker.rawEmbeddings,
            updateCount: speaker.updateCount,
            createdAt: speaker.createdAt,
            updatedAt: speaker.updatedAt
        )
    }

    /// Upsert a speaker - update if exists, insert if new
    ///
    /// - Parameters:
    ///   - id: The speaker ID
    ///   - currentEmbedding: The current embedding for the speaker
    ///   - duration: The total duration of speech
    ///   - rawEmbeddings: Raw embeddings for the speaker
    ///   - updateCount: Number of updates to this speaker
    ///   - createdAt: Creation timestamp
    ///   - updatedAt: Last update timestamp
    public func upsertSpeaker(
        id: String,
        currentEmbedding: [Float],
        duration: Float,
        rawEmbeddings: [RawEmbedding] = [],
        updateCount: Int = 1,
        createdAt: Date? = nil,
        updatedAt: Date? = nil
    ) {
        queue.sync(flags: .barrier) {
            let now = Date()

            if let existingSpeaker = speakerDatabase[id] {
                // Update existing speaker
                existingSpeaker.currentEmbedding = currentEmbedding
                existingSpeaker.duration = duration
                existingSpeaker.rawEmbeddings = rawEmbeddings
                existingSpeaker.updateCount = updateCount
                existingSpeaker.updatedAt = updatedAt ?? now
                // Keep original createdAt and name

                speakerDatabase[id] = existingSpeaker
                logger.info("Updated existing speaker: \(id)")
            } else {
                // Insert new speaker
                let newSpeaker = Speaker(
                    id: id,
                    name: id,  // Default name is the ID
                    currentEmbedding: currentEmbedding,
                    duration: duration,
                    createdAt: createdAt ?? now,
                    updatedAt: updatedAt ?? now
                )

                newSpeaker.rawEmbeddings = rawEmbeddings
                newSpeaker.updateCount = updateCount

                speakerDatabase[id] = newSpeaker

                // Update tracking for numeric IDs
                if let numericId = Int(id) {
                    highestSpeakerId = max(highestSpeakerId, numericId)
                    nextSpeakerId = max(nextSpeakerId, numericId + 1)
                }

                logger.info("Inserted new speaker: \(id)")
            }
        }
    }

    public func reset() {
        queue.sync(flags: .barrier) {
            speakerDatabase.removeAll()
            nextSpeakerId = 1
            highestSpeakerId = 0
            logger.info("Speaker database reset")
        }
    }
}
