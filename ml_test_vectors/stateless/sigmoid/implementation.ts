import type {
	StatelessDerivativeFunction,
	StatelessFunction,
	Tensor,
} from "../../core";
import { isScalar } from "../../utils";

export const forward: StatelessFunction<Tensor, Tensor> = (input) =>
	isScalar(input) ? 1 / (1 + Math.exp(-input)) : input.map(forward);

export const backward: StatelessDerivativeFunction = (
	input,
	output = forward(input),
) => {
	if (isScalar(output)) {
		return output * (1 - output);
	}
	if (isScalar(input)) {
		throw new Error("Mismatched input/output");
	}
	return output.map((o, i) => backward(input[i], o));
};
