import path from "path";

import { checkBackward, checkForward } from "../testutils";

import { backward, forward } from "./implementation";

const testVectorFilePath = path.join(__dirname, "test_vector.json");

describe("tanh", () => {
	// eslint-disable-next-line jest/expect-expect
	it("matches the test vector forward pass", () => {
		checkForward(testVectorFilePath, forward);
	});

	// eslint-disable-next-line jest/expect-expect
	it("matches the test vector backward pass", () => {
		checkBackward(testVectorFilePath, backward);
	});
});
