import type {
	Scalar,
	StatelessDerivativeFunction,
	StatelessFunction,
	Tensor,
} from "../../types";
import { isScalar, onesLike } from "../../utils";

export const forward: StatelessFunction<Tensor, Scalar> = (input) => {
	if (isScalar(input)) {
		return input;
	}
	return input.reduce<number>((subtotal, n) => subtotal + forward(n), 0);
};

export const backward: StatelessDerivativeFunction = (input) => onesLike(input);
