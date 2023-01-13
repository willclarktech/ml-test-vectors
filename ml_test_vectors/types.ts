export type Scalar = number;
export type Tensor = Scalar | readonly Tensor[];

export interface TestVector {
	readonly inputs: readonly Tensor[];
	readonly outputs: readonly Tensor[];
	readonly gradients: readonly Tensor[];
}

export type StatelessFunction = (input: Tensor) => Tensor;
export type StatelessDerivativeFunction = (
	input: Tensor,
	output?: Tensor,
) => Tensor;
