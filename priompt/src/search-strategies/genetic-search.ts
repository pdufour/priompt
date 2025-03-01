import { SearchResult } from './index';
import { PriomptTokenizer } from '../tokenizer';
import { PromptElement } from '../types.d';
import { countTokensApproxFast_UNSAFE, countTokensExact, renderWithLevelAndEarlyExitWithTokenEstimation, shouldPrintVerboseLogs } from '../lib';

/**
 * Genetic Algorithm Search Implementation
 *
 * This search strategy uses principles from genetic algorithms to efficiently find
 * the optimal priority level while minimizing token count operations.
 *
 * Key characteristics:
 * - Population-based approach with selection, crossover, and mutation
 * - Adaptive fitness evaluation to quickly narrow down the search space
 * - Handles non-monotonic relationships between priority levels and token counts
 * - Parallelizable evaluation for improved performance
 */

// Constants for the genetic algorithm
const POPULATION_SIZE = 8;
const MAX_GENERATIONS = 5;
const MUTATION_RATE = 0.2;
const CROSSOVER_RATE = 0.7;
// Added safety margin for token counts - helps account for differences between estimation and actual rendering
const TOKEN_SAFETY_MARGIN = 200;

export async function runGeneticSearch(
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

	if (shouldPrintVerboseLogs()) {
		console.debug(`[GeneticSearch] Starting search with ${sortedPriorityLevels.length} priority levels (token limit: ${usedTokenlimit}, search limit with margin: ${searchTokenLimit})`);
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
				console.debug(`[GeneticSearch] Cache hit for level ${level} (index ${index}): ${cachedCount} tokens (${isValid ? 'valid' : 'invalid'}, against search limit: ${searchTokenLimit})`);
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
					console.debug(`[GeneticSearch] Level ${level} (index ${index}) failed to render, treating as invalid`);
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
				}
			} else {
				if (!invalidIndices.includes(index)) {
					invalidIndices.push(index);
				}
			}

			if (shouldPrintVerboseLogs()) {
				console.debug(`[GeneticSearch] Tested level ${level} (index ${index}): ${totalTokens} tokens (${isValid ? 'valid' : 'invalid'}, against search limit: ${searchTokenLimit})`);
			}

			return {
				isValid,
				tokenCount: totalTokens
			};
		} catch (e) {
			if (shouldPrintVerboseLogs()) {
				console.debug(`[GeneticSearch] Error testing level ${level} (index ${index}): ${e instanceof Error ? e.message : String(e)}`);
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
			console.debug(`[GeneticSearch] ❌ Early exit: Even the highest priority level (${sortedPriorityLevels[highestIndex]}) exceeds token limit with ${highestResult.tokenCount} tokens`);
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
				console.debug(`[GeneticSearch] ✅ Quick exit: Lowest priority level (${sortedPriorityLevels[lowestIndex]}) is valid with ${lowestResult.tokenCount} tokens`);
				console.debug(`[GeneticSearch] Search completed in ${(endTime - startTime).toFixed(1)}ms`);
			}

			return {
				exclusiveLowerBound: -1,
				inclusiveUpperBound: lowestIndex,
				largestTokensSeen: largestTokenCountSeen
			};
		}
	}

	// Initialize population with strategic sampling
	let population = initializePopulation(sortedPriorityLevels.length);

	// Run the genetic algorithm
	for (let generation = 0; generation < MAX_GENERATIONS; generation++) {
		if (shouldPrintVerboseLogs()) {
			console.debug(`[GeneticSearch] Generation ${generation + 1}/${MAX_GENERATIONS}`);
		}

		// Evaluate fitness in parallel for efficiency
		const fitnessResults = await Promise.all(
			population.map(async (index) => {
				// Use exact token counting in the final generation to ensure accuracy
				const result = await testLevel(index, generation === MAX_GENERATIONS - 1);
				return {
					index,
					isValid: result.isValid,
					tokenCount: result.tokenCount,
					fitness: calculateFitness(index, result.isValid, result.tokenCount, searchTokenLimit)
				};
			})
		);

		// If we've found a valid solution, try to improve it
		if (validIndices.length > 0) {
			// Sort valid indices by priority level (ascending)
			validIndices.sort((a, b) => a - b);
			const bestValidIndex = validIndices[0];

			// If the best valid index is already the lowest priority level, we're done
			if (bestValidIndex === 0) {
				break;
			}

			// Focus search around the boundary
			population = focusSearchAroundBoundary(bestValidIndex, invalidIndices);
		} else {
			// Sort by fitness (descending)
			fitnessResults.sort((a, b) => b.fitness - a.fitness);

			// Select, crossover, and mutate
			population = evolvePopulation(
				fitnessResults.map(r => r.index),
				invalidIndices,
				validIndices,
				sortedPriorityLevels.length
			);
		}

		// Early termination if we've found an optimal solution
		if (validIndices.length > 0 && validIndices.includes(0)) {
			break;
		}
	}

	// Final verification of the best candidates
	if (validIndices.length === 0) {
		// No valid solution found at all, use the highest priority level as a fallback
		validIndices.push(highestIndex);

		if (shouldPrintVerboseLogs()) {
			console.debug(`[GeneticSearch] No valid solutions found, falling back to highest priority level ${sortedPriorityLevels[highestIndex]}`);
		}
	}

	// Find the best valid priority level (lowest priority level that's valid)
	validIndices.sort((a, b) => a - b);

	// Verify the top candidates using exact token counting to ensure they're truly valid
	let bestValidIndex = -1;

	// Test up to 3 best candidates with exact token counting
	const candidatesToTest = validIndices.slice(0, 3);

	if (shouldPrintVerboseLogs()) {
		console.debug(`[GeneticSearch] Final verification of ${candidatesToTest.length} candidates with exact token counting`);
	}

	for (const index of candidatesToTest) {
		// Force exact token counting for final verification and check against the actual token limit
		const result = await testLevel(index, true);

		// Now verify against the ACTUAL token limit, not the search limit with margin
		if (result.tokenCount <= usedTokenlimit) {
			bestValidIndex = index;
			if (shouldPrintVerboseLogs()) {
				console.debug(`[GeneticSearch] Verified candidate ${sortedPriorityLevels[index]} (index ${index}) is valid: ${result.tokenCount} tokens`);
			}
			break;
		} else {
			if (shouldPrintVerboseLogs()) {
				console.debug(`[GeneticSearch] Candidate ${sortedPriorityLevels[index]} (index ${index}) failed verification: ${result.tokenCount} tokens (exceeds ${usedTokenlimit})`);
			}
			// Remove from valid indices as it's actually invalid
			const idx = validIndices.indexOf(index);
			if (idx !== -1) {
				validIndices.splice(idx, 1);
			}
			invalidIndices.push(index);
		}
	}

	// If no candidate passed verification, fall back to the highest priority level
	if (bestValidIndex === -1) {
		bestValidIndex = highestIndex;
		if (shouldPrintVerboseLogs()) {
			console.debug(`[GeneticSearch] No candidate passed verification, falling back to highest priority level ${sortedPriorityLevels[highestIndex]}`);
		}
	}

	// Find the highest invalid level below the best valid level
	invalidIndices.sort((a, b) => b - a);
	let exclusiveLowerBound = -1;

	for (const index of invalidIndices) {
		if (index < bestValidIndex) {
			exclusiveLowerBound = index;
			break;
		}
	}

	if (shouldPrintVerboseLogs()) {
		const endTime = performance.now();
		console.debug(`[GeneticSearch] ✅ Search completed in ${(endTime - startTime).toFixed(1)}ms`);
		console.debug(`[GeneticSearch] Found best priority level: ${sortedPriorityLevels[bestValidIndex]} (index ${bestValidIndex})`);
		console.debug(`[GeneticSearch] Tested ${validIndices.length + invalidIndices.length} levels: ${validIndices.length} valid, ${invalidIndices.length} invalid`);
	}

	return {
		exclusiveLowerBound,
		inclusiveUpperBound: bestValidIndex,
		largestTokensSeen: largestTokenCountSeen
	};
}

/**
 * Initialize the population with strategic sampling across the range
 */
function initializePopulation(maxIndex: number): number[] {
	const population: number[] = [];

	// Always include the highest index
	population.push(maxIndex - 1);

	// Divide the range into segments and sample from each
	const segmentSize = Math.max(1, Math.floor(maxIndex / (POPULATION_SIZE - 1)));

	for (let i = 0; i < POPULATION_SIZE - 1; i++) {
		const index = Math.min(maxIndex - 1, Math.floor(i * segmentSize));
		if (!population.includes(index)) {
			population.push(index);
		}
	}

	// Fill remaining slots with random indices
	while (population.length < POPULATION_SIZE) {
		const randomIndex = Math.floor(Math.random() * maxIndex);
		if (!population.includes(randomIndex)) {
			population.push(randomIndex);
		}
	}

	return population;
}

/**
 * Calculate fitness based on proximity to the token limit
 */
function calculateFitness(index: number, isValid: boolean, tokenCount: number, usedTokenlimit: number): number {
	if (isValid) {
		// Valid solutions are prioritized, with lower indices being better
		// We want to maximize fitness for valid solutions with low indices
		const indexFitness = 1.0 - (index / 1000);
		const tokenProximity = 1.0 - Math.abs(tokenCount - usedTokenlimit) / usedTokenlimit;
		return 1000 + indexFitness + tokenProximity;
	} else {
		// Invalid solutions are ranked by how close they are to the token limit
		return 1.0 - Math.abs(tokenCount - usedTokenlimit) / usedTokenlimit;
	}
}

/**
 * Evolve the population using selection, crossover, and mutation
 */
function evolvePopulation(
	population: number[],
	invalidIndices: number[],
	validIndices: number[],
	maxIndex: number
): number[] {
	const newPopulation: number[] = [];

	// Elitism: keep the best 25% of the population
	const eliteCount = Math.max(1, Math.floor(POPULATION_SIZE * 0.25));
	for (let i = 0; i < eliteCount && i < population.length; i++) {
		newPopulation.push(population[i]);
	}

	// Fill the rest with crossover and mutation
	while (newPopulation.length < POPULATION_SIZE) {
		if (Math.random() < CROSSOVER_RATE && population.length >= 2) {
			// Select two parents using tournament selection
			const parent1 = tournamentSelection(population);
			const parent2 = tournamentSelection(population);

			// Crossover
			const child = crossover(parent1, parent2);

			// Mutation
			if (Math.random() < MUTATION_RATE) {
				mutate(child, maxIndex);
			}

			if (child >= 0 && child < maxIndex && !newPopulation.includes(child)) {
				newPopulation.push(child);
			}
		} else {
			// Mutation only
			const parent = tournamentSelection(population);
			const child = mutate(parent, maxIndex);

			if (child >= 0 && child < maxIndex && !newPopulation.includes(child)) {
				newPopulation.push(child);
			}
		}
	}

	// Ensure diversity by adding some exploration candidates
	if (validIndices.length > 0 && invalidIndices.length > 0) {
		// Find the boundary between valid and invalid
		const lowestValid = Math.min(...validIndices);
		const highestInvalid = Math.max(...invalidIndices.filter(i => i < lowestValid));

		// Add exploration candidates around the boundary
		const boundary = Math.floor((lowestValid + highestInvalid) / 2);
		if (!newPopulation.includes(boundary)) {
			newPopulation[newPopulation.length - 1] = boundary;
		}
	}

	return newPopulation;
}

/**
 * Tournament selection for parent selection
 */
function tournamentSelection(population: number[]): number {
	const tournamentSize = Math.max(2, Math.floor(population.length * 0.3));
	const tournament: number[] = [];

	// Select random individuals for the tournament
	for (let i = 0; i < tournamentSize; i++) {
		const randomIndex = Math.floor(Math.random() * population.length);
		tournament.push(population[randomIndex]);
	}

	// Return the best (we assume the population is already sorted by fitness)
	return tournament.reduce((best, current) =>
		population.indexOf(best) < population.indexOf(current) ? best : current
	);
}

/**
 * Crossover two parents to produce a child
 */
function crossover(parent1: number, parent2: number): number {
	// Simple arithmetic crossover
	return Math.floor((parent1 + parent2) / 2);
}

/**
 * Mutate an individual
 */
function mutate(individual: number, maxIndex: number): number {
	// Apply different mutation strategies
	const mutationStrategy = Math.random();

	if (mutationStrategy < 0.33) {
		// Small mutation: add/subtract a small random number
		const delta = Math.floor(Math.random() * 3) + 1;
		return Math.max(0, Math.min(maxIndex - 1, individual + (Math.random() < 0.5 ? delta : -delta)));
	} else if (mutationStrategy < 0.66) {
		// Medium mutation: shift by a percentage of the range
		const shift = Math.floor(maxIndex * 0.1 * (Math.random() < 0.5 ? 1 : -1));
		return Math.max(0, Math.min(maxIndex - 1, individual + shift));
	} else {
		// Large mutation: random jump
		return Math.floor(Math.random() * maxIndex);
	}
}

/**
 * Focus search around the current boundary between valid and invalid levels
 */
function focusSearchAroundBoundary(bestValidIndex: number, invalidIndices: number[]): number[] {
	const population: number[] = [];

	// Add the current best valid index
	population.push(bestValidIndex);

	// Find the highest invalid index below the best valid index
	const nearestInvalidIndices = invalidIndices
		.filter(idx => idx < bestValidIndex)
		.sort((a, b) => b - a);

	const highestInvalidIndex = nearestInvalidIndices.length > 0 ? nearestInvalidIndices[0] : 0;

	// Focus search around this boundary
	const searchRange = bestValidIndex - highestInvalidIndex;
	const stepSize = Math.max(1, Math.floor(searchRange / (POPULATION_SIZE - 2)));

	// Add candidates between the highest invalid and lowest valid
	for (let i = 1; i < POPULATION_SIZE - 1; i++) {
		const candidate = highestInvalidIndex + (i * stepSize);
		if (candidate < bestValidIndex && !population.includes(candidate)) {
			population.push(candidate);
		}
	}

	// Add some candidates slightly above and below the boundary for thoroughness
	if (!population.includes(Math.max(0, highestInvalidIndex - 1))) {
		population.push(Math.max(0, highestInvalidIndex - 1));
	}

	if (!population.includes(Math.max(0, highestInvalidIndex - 2))) {
		population.push(Math.max(0, highestInvalidIndex - 2));
	}

	// Fill remaining slots with values between 0 and highestInvalidIndex
	while (population.length < POPULATION_SIZE) {
		if (highestInvalidIndex > 0) {
			const randomIndex = Math.floor(Math.random() * highestInvalidIndex);
			if (!population.includes(randomIndex)) {
				population.push(randomIndex);
			}
		} else {
			break;
		}
	}

	return population;
}