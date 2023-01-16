import type {
	StatelessDerivativeFunction,
	StatelessFunction,
	Tensor,
	Vector,
} from "../../core";
import { isScalar, onesLike } from "../../utils";

export const forward: StatelessFunction<Tensor, Vector> = (input) =>
	isScalar(input)
		? [input]
		: input.reduce<Vector>((flattened, t) => [...flattened, ...forward(t)], []);

export const backward: StatelessDerivativeFunction = (input) => onesLike(input);
