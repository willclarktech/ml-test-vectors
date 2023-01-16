import type {
	StatelessDerivativeFunction,
	StatelessFunction,
	Tensor,
} from "../../core";
import { isScalar } from "../../utils";

export const forward: StatelessFunction<Tensor, Tensor> = (input) =>
	isScalar(input) ? Math.tanh(input) : input.map(forward);

export const backward: StatelessDerivativeFunction = (
	input,
	output = forward(input),
) => {
	if (isScalar(output)) {
		return 1 - output ** 2;
	}
	if (isScalar(input)) {
		throw new Error("Mismatched input/output");
	}
	return output.map((o, i) => backward(input[i], o));
};
