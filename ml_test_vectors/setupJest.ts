/* eslint-disable @typescript-eslint/no-namespace, functional/no-method-signature, functional/prefer-readonly-type */
import { expect } from "@jest/globals";
import type { MatcherFunction } from "expect";

import type { Tensor } from "./core";

const isCloseTo = (a: Tensor, b: Tensor, numDigits: number): boolean => {
	if (typeof a === "number") {
		if (typeof b !== "number") {
			throw new Error(`Mismatched tensors: ${a}, ${JSON.stringify(b)}`);
		}
		return Math.abs(a - b) < 10 ** -numDigits / 2;
	}
	if (typeof b === "number") {
		throw new Error(`Mismatched tensors: ${JSON.stringify(a)}, ${b}`);
	}
	return a.every((n: Tensor, i: number): boolean =>
		isCloseTo(n, b[i], numDigits),
	);
};

const toBeCloseTo: MatcherFunction<[value: Tensor]> =
	// `floor` and `ceiling` get types from the line above
	// it is recommended to type them as `unknown` and to validate the values
	function (actual: unknown, value: Tensor, numDigits = 2) {
		if (typeof actual !== "number" && !Array.isArray(actual)) {
			throw new Error("Must pass a number or Tensor for comparison");
		}

		const pass = isCloseTo(actual, value, numDigits);
		if (pass) {
			return {
				message: () =>
					`expected ${this.utils.printReceived(
						actual,
					)} not to be close to ${this.utils.printExpected(value)}`,
				pass: true,
			};
		} else {
			return {
				message: () =>
					`expected ${this.utils.printReceived(
						actual,
					)} to be close to ${this.utils.printExpected(value)}`,
				pass: false,
			};
		}
	};

expect.extend({
	toBeCloseTo,
});

declare module "expect" {
	interface AsymmetricMatchers {
		toBeCloseTo(value: Tensor): void;
	}
	interface Matchers<R> {
		toBeCloseTo(value: Tensor): R;
	}
}

declare global {
	namespace jest {
		interface Matchers<R> {
			toBeCloseTo(value: Tensor): R;
		}
	}
}
