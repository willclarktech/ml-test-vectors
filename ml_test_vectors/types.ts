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

export type StatelessFunction<
	I extends Tensor = Tensor,
	R extends Tensor = Tensor,
> = (input: I) => R;
export type StatelessDerivativeFunction<
	I extends Tensor = Tensor,
	O extends Tensor = Tensor,
	R extends Tensor = Tensor,
> = (input: I, output?: O) => R;
