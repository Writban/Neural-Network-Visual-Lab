export type ActivationName = "tanh" | "relu" | "sigmoid";
export type DatasetKind = "xor" | "circles" | "moons";

export type Sample = { x: number; y: number; label: 0 | 1 };
export type DenseLayer = { weights: number[][]; biases: number[] };
export type NeuralNetwork = { layers: DenseLayer[] };
export type ForwardPass = {
  output: number;
  activations: number[][];
  weightedInputs: number[][];
};
export type Metrics = { loss: number; accuracy: number };

export const architectureOptions = [
  { label: "Compact", value: "3", layers: [3] },
  { label: "Balanced", value: "6-4", layers: [6, 4] },
  { label: "Deep", value: "8-6-4", layers: [8, 6, 4] },
] as const;

export function architectureFromValue(value: string): number[] {
  const preset = architectureOptions.find((option) => option.value === value);
  if (preset) return preset.layers.slice();

  const customLayers = value
    .split("-")
    .map((size) => Number.parseInt(size, 10))
    .filter((size) => Number.isFinite(size) && size >= 1 && size <= 12)
    .slice(0, 4);
  return customLayers.length > 0
    ? customLayers
    : architectureOptions[1].layers.slice();
}

export function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  const first = Math.max(rng(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * rng());
}

export function generateDataset(
  kind: DatasetKind,
  seed: number,
  count = 180,
): Sample[] {
  const rng = createRng(seed);
  const samples: Sample[] = [];

  if (kind === "moons") {
    for (let index = 0; index < count; index += 1) {
      const label = (index % 2) as 0 | 1;
      const angle = rng() * Math.PI;
      const noiseX = gaussian(rng) * 0.07;
      const noiseY = gaussian(rng) * 0.07;
      samples.push(
        label === 0
          ? {
              x: Math.cos(angle) * 0.65 + noiseX,
              y: Math.sin(angle) * 0.65 - 0.23 + noiseY,
              label,
            }
          : {
              x: 0.38 - Math.cos(angle) * 0.65 + noiseX,
              y: 0.23 - Math.sin(angle) * 0.65 + noiseY,
              label,
            },
      );
    }
    return samples;
  }

  for (let index = 0; index < count; index += 1) {
    const x = rng() * 1.8 - 0.9;
    const y = rng() * 1.8 - 0.9;
    if (kind === "circles") {
      const radius = Math.sqrt(x * x + y * y) + gaussian(rng) * 0.035;
      samples.push({ x, y, label: radius > 0.55 ? 1 : 0 });
    } else {
      const noisyX = x + gaussian(rng) * 0.04;
      const noisyY = y + gaussian(rng) * 0.04;
      samples.push({
        x: noisyX,
        y: noisyY,
        label: (noisyX > 0) !== (noisyY > 0) ? 1 : 0,
      });
    }
  }
  return samples;
}

function activate(value: number, activation: ActivationName): number {
  if (activation === "relu") return Math.max(0, value);
  if (activation === "sigmoid") return 1 / (1 + Math.exp(-value));
  return Math.tanh(value);
}

function activationDerivative(
  weightedInput: number,
  activatedValue: number,
  activation: ActivationName,
): number {
  if (activation === "relu") return weightedInput > 0 ? 1 : 0;
  if (activation === "sigmoid") return activatedValue * (1 - activatedValue);
  return 1 - activatedValue * activatedValue;
}

function outputActivation(value: number): number {
  if (value >= 0) {
    const exponential = Math.exp(-value);
    return 1 / (1 + exponential);
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

export function createNetwork(hiddenLayers: number[], seed: number): NeuralNetwork {
  const rng = createRng(seed);
  const sizes = [2, ...hiddenLayers, 1];
  const layers: DenseLayer[] = [];
  for (let layerIndex = 1; layerIndex < sizes.length; layerIndex += 1) {
    const inputSize = sizes[layerIndex - 1];
    const outputSize = sizes[layerIndex];
    const limit = Math.sqrt(6 / (inputSize + outputSize));
    layers.push({
      weights: Array.from({ length: outputSize }, () =>
        Array.from({ length: inputSize }, () => (rng() * 2 - 1) * limit),
      ),
      biases: Array.from({ length: outputSize }, () => 0),
    });
  }
  return { layers };
}

export function forward(
  network: NeuralNetwork,
  input: [number, number],
  activation: ActivationName,
): ForwardPass {
  const activations: number[][] = [[input[0], input[1]]];
  const weightedInputs: number[][] = [];
  network.layers.forEach((layer, layerIndex) => {
    const previous = activations[activations.length - 1];
    const weighted = layer.weights.map(
      (weights, neuronIndex) =>
        weights.reduce(
          (sum, weight, inputIndex) => sum + weight * previous[inputIndex],
          layer.biases[neuronIndex],
        ),
    );
    const isOutputLayer = layerIndex === network.layers.length - 1;
    const next = weighted.map((value) =>
      isOutputLayer ? outputActivation(value) : activate(value, activation),
    );
    weightedInputs.push(weighted);
    activations.push(next);
  });
  return {
    output: activations[activations.length - 1][0],
    activations,
    weightedInputs,
  };
}

export function evaluate(
  network: NeuralNetwork,
  samples: Sample[],
  activation: ActivationName,
): Metrics {
  let loss = 0;
  let correct = 0;
  samples.forEach((sample) => {
    const prediction = forward(network, [sample.x, sample.y], activation).output;
    const clipped = Math.min(1 - 1e-7, Math.max(1e-7, prediction));
    loss +=
      -sample.label * Math.log(clipped) -
      (1 - sample.label) * Math.log(1 - clipped);
    if ((prediction >= 0.5 ? 1 : 0) === sample.label) correct += 1;
  });
  return { loss: loss / samples.length, accuracy: correct / samples.length };
}

export function trainEpoch(
  network: NeuralNetwork,
  samples: Sample[],
  activation: ActivationName,
  learningRate: number,
): Metrics {
  const weightGradients = network.layers.map((layer) =>
    layer.weights.map((weights) => weights.map(() => 0)),
  );
  const biasGradients = network.layers.map((layer) => layer.biases.map(() => 0));

  samples.forEach((sample) => {
    const pass = forward(network, [sample.x, sample.y], activation);
    const deltas: number[][] = Array.from(
      { length: network.layers.length },
      () => [],
    );
    const lastIndex = network.layers.length - 1;
    deltas[lastIndex] = [pass.output - sample.label];
    for (let layerIndex = lastIndex - 1; layerIndex >= 0; layerIndex -= 1) {
      const nextLayer = network.layers[layerIndex + 1];
      deltas[layerIndex] = network.layers[layerIndex].biases.map((_, neuron) => {
        const propagated = nextLayer.weights.reduce(
          (sum, weights, nextNeuron) =>
            sum + weights[neuron] * deltas[layerIndex + 1][nextNeuron],
          0,
        );
        return (
          propagated *
          activationDerivative(
            pass.weightedInputs[layerIndex][neuron],
            pass.activations[layerIndex + 1][neuron],
            activation,
          )
        );
      });
    }

    network.layers.forEach((layer, layerIndex) => {
      const previousActivations = pass.activations[layerIndex];
      layer.weights.forEach((weights, neuronIndex) => {
        weights.forEach((_, inputIndex) => {
          weightGradients[layerIndex][neuronIndex][inputIndex] +=
            deltas[layerIndex][neuronIndex] * previousActivations[inputIndex];
        });
        biasGradients[layerIndex][neuronIndex] += deltas[layerIndex][neuronIndex];
      });
    });
  });

  const scale = learningRate / samples.length;
  network.layers.forEach((layer, layerIndex) => {
    layer.weights.forEach((weights, neuronIndex) => {
      weights.forEach((_, inputIndex) => {
        layer.weights[neuronIndex][inputIndex] -=
          scale * weightGradients[layerIndex][neuronIndex][inputIndex];
      });
      layer.biases[neuronIndex] -= scale * biasGradients[layerIndex][neuronIndex];
    });
  });
  return evaluate(network, samples, activation);
}
