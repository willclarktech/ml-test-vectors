import fs from "fs";

import type {
	StatelessDerivativeFunction,
	StatelessFunction,
	TestVector,
} from "../core";

export const loadTestVector = (filePath: string): TestVector => {
	const contents = fs.readFileSync(filePath, "utf8");
	// TODO: Validate shape
	return JSON.parse(contents) as TestVector;
};

export const checkForward = (filePath: string, fn: StatelessFunction): void => {
	const testVector = loadTestVector(filePath);
	testVector.inputs.forEach((input, i) => {
		expect(fn(input)).toBeCloseTo(testVector.outputs[i]);
	});
};

export const checkBackward = (
	filePath: string,
	fn: StatelessDerivativeFunction,
): void => {
	const testVector = loadTestVector(filePath);
	testVector.inputs.forEach((input, i) => {
		expect(fn(input, testVector.outputs[i])).toBeCloseTo(
			testVector.gradients[i],
		);
	});
};
