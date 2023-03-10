import type {
	Scalar,
	StatelessDerivativeFunction,
	StatelessFunction,
	Tensor,
} from "../../core";
import { isScalar, onesLike } from "../../utils";

export const forward: StatelessFunction<Tensor, Scalar> = (input) =>
	isScalar(input)
		? input
		: input.reduce<number>((subtotal, n) => subtotal + forward(n), 0);

export const backward: StatelessDerivativeFunction = (input) => onesLike(input);
