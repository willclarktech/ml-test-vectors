import type {
	Scalar,
	StatelessDerivativeFunction,
	StatelessFunction,
	Tensor,
	Vector,
} from "../../core";
import { isScalar, isVector, onesLike } from "../../utils";

export const forward: StatelessFunction<Tensor, Scalar | Vector> = (input) => {
	if (isScalar(input) || isVector(input)) {
		return input;
	}
	return input.reduce<Scalar | Vector>((flattened, t) => {
		if (isScalar(flattened)) {
			throw new Error("Invalid tensor");
		}
		const flattenedT = forward(t);
		if (isScalar(flattenedT)) {
			throw new Error("Invalid tensor");
		}
		return [...flattened, ...flattenedT];
	}, []);
};

export const backward: StatelessDerivativeFunction = (input) => onesLike(input);
