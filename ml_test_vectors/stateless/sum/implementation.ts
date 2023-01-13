import type { Scalar, Tensor } from "../../types";
import { isScalar, onesLike } from "../../utils";

export const forward = (input: Tensor): Scalar => {
	if (isScalar(input)) {
		return input;
	}
	return input.reduce<number>((subtotal, n) => subtotal + forward(n), 0);
};

export const backward = (input: Tensor, output?: Tensor): Tensor =>
	onesLike(input);
