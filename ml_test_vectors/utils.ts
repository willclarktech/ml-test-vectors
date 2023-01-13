import type { Scalar, Tensor } from "./core";

export const isScalar = (tensor: Tensor): tensor is Scalar =>
	typeof tensor === "number";

export const onesLike = (tensor: Tensor): Tensor => {
	if (isScalar(tensor)) {
		return 1;
	}
	return tensor.map(onesLike);
};
