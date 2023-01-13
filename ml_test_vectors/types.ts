export type Scalar = number;
// TODO: Define more precisely with max dimensions?
// eg
// Tensor0 = Scalar
// Tensor1 = readonly Tensor0[]
// Tensor2 = readonly Tensor1[]
// Tensor = Tensor0 | Tensor1 | Tensor2...
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
