import { SearchResult } from './index';
import { PriomptTokenizer } from '../tokenizer';
import { PromptElement } from '../types.d';
import { countTokensApproxFast_UNSAFE, countTokensExact, renderWithLevelAndEarlyExitWithTokenEstimation, shouldPrintVerboseLogs } from '../lib';

/**
 * Ant Colony Optimization Search Implementation
 *
 * This search strategy uses principles from ant colony optimization to efficiently
 * find the optimal priority level while minimizing token count operations.
 *
 * Key characteristics:
 * - Based on the foraging behavior of ants
 * - Uses pheromone trails to guide search towards promising regions
 * - Balances exploration and exploitation through pheromone evaporation
 * - Well-suited for dynamic, changing token distributions
 */

// Constants for the ant colony algorithm
const NUM_ANTS = 8;
const MAX_ITERATIONS = 5;
const PHEROMONE_EVAPORATION_RATE = 0.1;
const PHEROMONE_DEPOSIT_FACTOR = 1.0;
const ALPHA = 1.0; // Pheromone importance
const BETA = 2.0;  // Heuristic importance
// Added safety margin for token counts - helps account for differences between estimation and actual rendering
const TOKEN_SAFETY_MARGIN = 200;

export async function runAntColonySearch(
	elem: PromptElement,
	sortedPriorityLevels: number[],
	tokenizer: PriomptTokenizer,
	tokenLimit: number,
	usedTokenlimit: number,
	countTokensFast_UNSAFE: boolean | undefined,
	lastMessageIsIncomplete: boolean | undefined,
	initialLargestTokenCountSeen: number = 0,
	initialTokenCounts: Map<number, number> = new Map()
): Promise<SearchResult> {
	// Early exit for empty array
	if (sortedPriorityLevels.length === 0) {
		return {
			exclusiveLowerBound: -1,
			inclusiveUpperBound: 0,
			largestTokensSeen: initialLargestTokenCountSeen
		};
	}

	// Adjust token limit with safety margin for search process
	const searchTokenLimit = Math.max(0, usedTokenlimit - TOKEN_SAFETY_MARGIN);

	let largestTokenCountSeen = initialLargestTokenCountSeen;
	const tokenCounts = new Map<number, number>(initialTokenCounts);
	const startTime = performance.now();

	// Track valid and invalid levels
	const validIndices: number[] = [];
	const invalidIndices: number[] = [];

	// Initialize pheromone trails and heuristic information
	const pheromones = new Array(sortedPriorityLevels.length).fill(1.0);
	const heuristic = new Array(sortedPriorityLevels.length).fill(1.0);

	if (shouldPrintVerboseLogs()) {
		console.debug(`[AntColonySearch] Starting search with ${sortedPriorityLevels.length} priority levels (token limit: ${usedTokenlimit}, search limit with margin: ${searchTokenLimit})`);
	}

	// Helper function to test a level and determine if it's valid
	async function testLevel(index: number, forceExact: boolean = false): Promise<{ isValid: boolean; tokenCount: number }> {
		const level = sortedPriorityLevels[index];

		// Check cache first
		if (tokenCounts.has(level)) {
			const cachedCount = tokenCounts.get(level)!;
			// When using cached values, still apply safety margin
			const isValid = cachedCount <= searchTokenLimit;

			if (shouldPrintVerboseLogs()) {
				console.debug(`[AntColonySearch] Cache hit for level ${level} (index ${index}): ${cachedCount} tokens (${isValid ? 'valid' : 'invalid'}, against search limit: ${searchTokenLimit})`);
			}

			return {
				isValid,
				tokenCount: cachedCount
			};
		}

		try {
			const prompt = renderWithLevelAndEarlyExitWithTokenEstimation(elem, level, tokenizer, tokenLimit);

			// If rendering failed or returned null prompt, consider invalid
			if (!prompt || !prompt.prompt) {
				if (shouldPrintVerboseLogs()) {
					console.debug(`[AntColonySearch] Level ${level} (index ${index}) failed to render, treating as invalid`);
				}
				tokenCounts.set(level, Number.MAX_SAFE_INTEGER);
				return {
					isValid: false,
					tokenCount: Number.MAX_SAFE_INTEGER
				};
			}

			let tokens: number;

			// Use exact counting when forced or for final verification
			if (forceExact || !countTokensFast_UNSAFE) {
				tokens = await countTokensExact(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete });
			} else {
				tokens = await countTokensApproxFast_UNSAFE(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete });
			}

			// Add any empty token count to the total
			const totalTokens = tokens + (prompt.emptyTokenCount || 0);

			// Update cache and largest seen
			tokenCounts.set(level, totalTokens);
			if (totalTokens > largestTokenCountSeen) {
				largestTokenCountSeen = totalTokens;
			}

			// Compare against the search token limit with safety margin
			const isValid = totalTokens <= searchTokenLimit;

			if (isValid) {
				if (!validIndices.includes(index)) {
					validIndices.push(index);
					// Update heuristic information - valid levels get better heuristic values
					// Better heuristic for higher token counts within valid range
					heuristic[index] = (totalTokens + 1) / (searchTokenLimit + 1);
				}
			} else {
				if (!invalidIndices.includes(index)) {
					invalidIndices.push(index);
					// Update heuristic information - invalid levels get worse heuristic values
					// Less penalty for being closer to the limit
					heuristic[index] = searchTokenLimit / (totalTokens + 1);
				}
			}

			if (shouldPrintVerboseLogs()) {
				console.debug(`[AntColonySearch] Tested level ${level} (index ${index}): ${totalTokens} tokens (${isValid ? 'valid' : 'invalid'}, against search limit: ${searchTokenLimit})`);
			}

			return {
				isValid,
				tokenCount: totalTokens
			};
		} catch (e) {
			if (shouldPrintVerboseLogs()) {
				console.debug(`[AntColonySearch] Error testing level ${level} (index ${index}): ${e instanceof Error ? e.message : String(e)}`);
			}

			// Assume invalid on error
			if (!invalidIndices.includes(index)) {
				invalidIndices.push(index);
			}

			return {
				isValid: false,
				tokenCount: largestTokenCountSeen
			};
		}
	}

	// First, test the highest priority level to see if it's already over the limit
	// Always use exact token counting for the highest priority level
	const highestIndex = sortedPriorityLevels.length - 1;
	const highestResult = await testLevel(highestIndex, true);

	if (!highestResult.isValid) {
		if (shouldPrintVerboseLogs()) {
			console.debug(`[AntColonySearch] ❌ Early exit: Even the highest priority level (${sortedPriorityLevels[highestIndex]}) exceeds token limit with ${highestResult.tokenCount} tokens`);
		}

		throw new Error(`Base prompt estimated token count is ${highestResult.tokenCount} with 0 tokens reserved, which is higher than the limit ${usedTokenlimit}. This is probably a bug in the prompt — please add some priority levels to fix this.`);
	}

	// Then test the lowest priority level
	if (highestIndex > 0) {
		const lowestIndex = 0;
		// For the lowest level, also use exact counting since it's an important boundary
		const lowestResult = await testLevel(lowestIndex, true);

		if (lowestResult.isValid) {
			// Lowest level is valid, we're done!
			if (shouldPrintVerboseLogs()) {
				const endTime = performance.now();
				console.debug(`[AntColonySearch] ✅ Quick exit: Lowest priority level (${sortedPriorityLevels[lowestIndex]}) is valid with ${lowestResult.tokenCount} tokens`);
				console.debug(`[AntColonySearch] Search completed in ${(endTime - startTime).toFixed(1)}ms`);
			}

			return {
				exclusiveLowerBound: -1,
				inclusiveUpperBound: lowestIndex,
				largestTokensSeen: largestTokenCountSeen
			};
		}
	}

	// Run the Ant Colony Optimization algorithm
	for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
		if (shouldPrintVerboseLogs()) {
			console.debug(`[AntColonySearch] Iteration ${iteration + 1}/${MAX_ITERATIONS}`);
		}

		// For the first iteration, test a few points across the range to better understand the space
		if (iteration === 0 && validIndices.length === 1 && validIndices[0] === highestIndex) {
			// Test a few strategic points across the index range to better understand the space
			const pointsToTest = [
				0, // Lowest priority
				Math.floor(highestIndex * 0.25),
				Math.floor(highestIndex * 0.5),
				Math.floor(highestIndex * 0.75)
			];

			await Promise.all(pointsToTest.map(idx => testLevel(idx, false)));
		}

		// Initialize ant positions - some random, some at boundaries
		const antPositions: number[] = [];
		for (let i = 0; i < NUM_ANTS; i++) {
			if (i < 2 && validIndices.length > 0) {
				// Place some ants at known valid positions
				antPositions.push(validIndices[Math.floor(Math.random() * validIndices.length)]);
			} else if (i < 4 && invalidIndices.length > 0) {
				// Place some ants at known invalid positions
				antPositions.push(invalidIndices[Math.floor(Math.random() * invalidIndices.length)]);
			} else {
				// Place remaining ants randomly, with bias towards unexplored regions
				antPositions.push(selectPositionByPheromone(pheromones, heuristic, sortedPriorityLevels.length));
			}
		}

		// Evaluate ants in parallel for efficiency
		const antResults = await Promise.all(
			antPositions.map(async (position) => {
				// Use exact token counting in the final iteration for accuracy
				const result = await testLevel(position, iteration === MAX_ITERATIONS - 1);
				return {
					position,
					isValid: result.isValid,
					tokenCount: result.tokenCount,
					quality: calculateQuality(position, result.isValid, result.tokenCount, searchTokenLimit)
				};
			})
		);

		// Update pheromones
		updatePheromones(pheromones, antResults, sortedPriorityLevels.length);

		// If we've found a valid solution with the lowest priority, we're done
		if (validIndices.length > 0 && validIndices.includes(0)) {
			break;
		}

		// If we have both valid and invalid indices, we can binary search the boundary
		// This helps us converge faster to the exact boundary between valid and invalid
		if (validIndices.length > 0 && invalidIndices.length > 0) {
			// Sort indices
			validIndices.sort((a, b) => a - b);
			invalidIndices.sort((a, b) => a - b);

			const lowestValidIndex = validIndices[0];
			const highestInvalidIndex = findHighestInvalidIndex(invalidIndices, lowestValidIndex);

			// If there's a gap between the highest invalid and lowest valid, search the gap
			if (highestInvalidIndex !== -1 && lowestValidIndex - highestInvalidIndex > 1) {
				const middle = Math.floor((highestInvalidIndex + lowestValidIndex) / 2);
				await testLevel(middle, iteration === MAX_ITERATIONS - 1);
			}

			// Increase pheromones in promising regions close to the boundary
			for (let i = Math.max(0, lowestValidIndex - 3); i <= lowestValidIndex + 1; i++) {
				if (i < sortedPriorityLevels.length) {
					pheromones[i] *= 1.5;
				}
			}

			// In later iterations, focus more on the exact boundary
			if (iteration >= MAX_ITERATIONS - 2) {
				// Test positions one by one near the boundary
				for (let i = Math.max(0, highestInvalidIndex); i <= lowestValidIndex; i++) {
					if (!validIndices.includes(i) && !invalidIndices.includes(i)) {
						await testLevel(i, true); // Use exact counting for boundary tests
					}
				}
			}
		}

		// Always ensure we test a range of indices near any promising areas
		// This helps avoid getting stuck in local optima
		if (iteration > 0 && validIndices.length > 0) {
			const minValid = Math.min(...validIndices);
			// Test boundary area with finer granularity in later iterations
			for (let i = Math.max(0, minValid - 5); i < minValid; i++) {
				if (!validIndices.includes(i) && !invalidIndices.includes(i)) {
					await testLevel(i, iteration === MAX_ITERATIONS - 1);
				}
			}
		}
	}

	// As a final step, verify the boundary with exact token counting
	// First ensure we have at least some valid solutions
	if (validIndices.length === 0) {
		// If no valid solutions below the highest level, use the highest level
		return {
			exclusiveLowerBound: highestIndex - 1,
			inclusiveUpperBound: highestIndex,
			largestTokensSeen: largestTokenCountSeen
		};
	}

	// Sort valid indices and find the boundary
	validIndices.sort((a, b) => a - b);
	const lowestValidIndex = validIndices[0];

	// For any invalid indices close to the boundary, verify them with exact counting
	const indicesToVerify = invalidIndices.filter(idx =>
		idx >= lowestValidIndex - 3 && idx < lowestValidIndex
	);

	// Verify these with exact token counting
	for (const idx of indicesToVerify) {
		await testLevel(idx, true);
	}

	// Also verify the lowest valid index and its neighbors with exact counting
	await testLevel(lowestValidIndex, true);
	if (lowestValidIndex > 0) {
		await testLevel(lowestValidIndex - 1, true);
	}
	if (lowestValidIndex < sortedPriorityLevels.length - 1) {
		await testLevel(lowestValidIndex + 1, true);
	}

	// Re-check which is the lowest valid index after exact verification
	validIndices.sort((a, b) => a - b);
	const verifiedLowestValidIndex = validIndices[0];
	const highestInvalidIndex = findHighestInvalidIndex(invalidIndices, verifiedLowestValidIndex);

	const endTime = performance.now();
	if (shouldPrintVerboseLogs()) {
		console.debug(`[AntColonySearch] ✅ Found solution: Lowest valid priority level is ${sortedPriorityLevels[verifiedLowestValidIndex]} (index ${verifiedLowestValidIndex})`);
		console.debug(`[AntColonySearch] Search completed in ${(endTime - startTime).toFixed(1)}ms`);
	}

	return {
		exclusiveLowerBound: highestInvalidIndex,
		inclusiveUpperBound: verifiedLowestValidIndex,
		largestTokensSeen: largestTokenCountSeen
	};
}

/**
 * Select a position for an ant based on pheromone levels and heuristic information
 */
function selectPositionByPheromone(pheromones: number[], heuristic: number[], maxIndex: number): number {
	// Calculate probabilities for each position
	const probabilities: { index: number; probability: number }[] = [];
	let totalProbability = 0;

	for (let i = 0; i < maxIndex; i++) {
		const probability = Math.pow(pheromones[i], ALPHA) * Math.pow(heuristic[i], BETA);
		probabilities.push({ index: i, probability });
		totalProbability += probability;
	}

	// Normalize probabilities
	if (totalProbability > 0) {
		for (let i = 0; i < probabilities.length; i++) {
			probabilities[i].probability /= totalProbability;
		}
	} else {
		// If all probabilities are zero, use uniform distribution
		for (let i = 0; i < probabilities.length; i++) {
			probabilities[i].probability = 1 / maxIndex;
		}
	}

	// Select position using roulette wheel selection
	const random = Math.random();
	let cumulativeProbability = 0;

	for (const { index, probability } of probabilities) {
		cumulativeProbability += probability;
		if (random <= cumulativeProbability) {
			return index;
		}
	}

	// Fallback - return random index
	return Math.floor(Math.random() * maxIndex);
}

/**
 * Calculate the quality of a solution found by an ant
 */
function calculateQuality(
	index: number,
	isValid: boolean,
	tokenCount: number,
	tokenLimit: number
): number {
	if (isValid) {
		// Valid solutions: prefer lower indices (lower priority levels)
		// and solutions closer to the token limit (more efficient use of tokens)
		// The closer to the token limit, the higher quality the solution
		const tokenEfficiency = Math.pow(tokenCount / (tokenLimit + 1), 2); // Squared to emphasize high token utilization
		const priorityValue = 1 - (index / 1000); // Favor lower priority indices

		// Heavily weight towards using more tokens, with secondary focus on lower priority
		return 0.2 + (tokenEfficiency * 0.7) + (priorityValue * 0.1);
	} else {
		// Invalid solutions: give higher value to those just above the limit
		// This helps focus the search near the boundary
		const overageRatio = tokenCount / (tokenLimit + 1);
		if (overageRatio <= 1.2) { // Within 20% of the limit
			// Solutions just above the limit are more interesting for boundary exploration
			return 0.1 * (1 - (overageRatio - 1));
		} else {
			// Solutions far above the limit are less valuable
			return 0.01;
		}
	}
}

/**
 * Update pheromone trails based on the solutions found by ants
 */
function updatePheromones(
	pheromones: number[],
	antResults: Array<{ position: number; isValid: boolean; quality: number }>,
	maxIndex: number
): void {
	// Evaporate pheromones
	for (let i = 0; i < pheromones.length; i++) {
		pheromones[i] *= (1 - PHEROMONE_EVAPORATION_RATE);
	}

	// Add new pheromones based on solution quality
	for (const result of antResults) {
		if (result.isValid) {
			// Deposit more pheromones for valid solutions
			pheromones[result.position] += PHEROMONE_DEPOSIT_FACTOR * result.quality;

			// Also deposit some pheromones in the neighborhood (with decay)
			const radius = 2;
			for (let i = 1; i <= radius; i++) {
				const lowerNeighbor = result.position - i;
				const upperNeighbor = result.position + i;

				if (lowerNeighbor >= 0) {
					pheromones[lowerNeighbor] += PHEROMONE_DEPOSIT_FACTOR * result.quality * (1 - 0.3 * i);
				}

				if (upperNeighbor < maxIndex) {
					pheromones[upperNeighbor] += PHEROMONE_DEPOSIT_FACTOR * result.quality * (1 - 0.4 * i);
				}
			}
		} else {
			// Add a small amount of pheromones even for invalid solutions to encourage exploration
			pheromones[result.position] += 0.1 * PHEROMONE_DEPOSIT_FACTOR * result.quality;
		}
	}
}

/**
 * Find the highest invalid index that is less than the lowest valid index
 */
function findHighestInvalidIndex(invalidIndices: number[], lowestValidIndex: number): number {
	return invalidIndices
		.filter(idx => idx < lowestValidIndex)
		.reduce((highest, current) => Math.max(highest, current), -1);
}