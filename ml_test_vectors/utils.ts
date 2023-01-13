import type { Scalar, Tensor, Vector } from "./core";

export const isScalar = (tensor: Tensor): tensor is Scalar =>
	typeof tensor === "number";

export const isVector = (tensor: Tensor): tensor is Vector =>
	Array.isArray(tensor) &&
	(tensor.length === 0 || isScalar((tensor as readonly Tensor[])[0]));

export const onesLike = (tensor: Tensor): Tensor => {
	if (isScalar(tensor)) {
		return 1;
	}
	return tensor.map(onesLike);
};
