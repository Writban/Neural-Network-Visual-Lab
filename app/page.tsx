"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type ActivationName,
  architectureFromValue,
  architectureOptions,
  createNetwork,
  createRng,
  evaluate,
  forward,
  generateDataset,
  type DatasetKind,
  type Metrics,
  type NeuralNetwork,
  type Sample,
  trainEpoch,
} from "./neural-core";
import NeuralWorkbench from "./neural-workbench";

type ExperimentConfig = {
  architecture: string;
  activation: ActivationName;
  learningRate: number;
};

type Probe = { x: number; y: number; probability: number };

const initialConfigs: [ExperimentConfig, ExperimentConfig] = [
  { architecture: "3", activation: "tanh", learningRate: 0.03 },
  { architecture: "6-4", activation: "relu", learningRate: 0.08 },
];

const datasetCopy: Record<DatasetKind, { title: string; note: string }> = {
  xor: {
    title: "XOR quadrants",
    note: "A linear boundary cannot separate the alternating quadrants.",
  },
  circles: {
    title: "Nested circles",
    note: "The network must wrap a boundary around the centre class.",
  },
  moons: {
    title: "Interleaving moons",
    note: "Curved, offset classes expose differences between configurations.",
  },
};

type TutorialActionType =
  | "probe"
  | "train-step"
  | "change-learning-rate"
  | "change-activation"
  | "change-architecture"
  | "change-noise"
  | "change-seed"
  | "change-architecture-b"
  | "change-activation-b"
  | "train-both"
  | "enable-mode";

type CourseSegment = {
  id: string;
  chapter: 1 | 2 | 3 | 4 | 5;
  chapterTitle: string;
  eyebrow: string;
  title: string;
  body: string;
  why: string;
  technical: string;
  tryThis: string;
  bonusFact: string;
  target: string;
  action?: { type: TutorialActionType; prompt: string };
  quiz?: {
    question: string;
    choices: string[];
    answer: number;
    explanation: string;
  };
};

type TutorialEvent = {
  type:
    | "probe"
    | "train-step"
    | "train-start"
    | "change-learning-rate"
    | "change-activation"
    | "change-architecture"
    | "change-noise"
    | "change-seed"
    | "enable-mode";
  experiment?: "A" | "B";
  feedback: string;
};

const chapterOneSegments: CourseSegment[] = [
  {
    id: "classification",
    chapter: 1,
    chapterTitle: "Understanding predictions",
    eyebrow: "What is the task?",
    title: "Classification means choosing between categories",
    body: "This lab gives a neural network examples from two groups and asks it to learn how to tell those groups apart. Cream and teal are the two known answers.",
    why: "Real classifiers use the same idea for tasks such as identifying a type of flower or sorting a message. Dots make the problem visible without assuming any previous knowledge.",
    technical: "This is supervised binary classification: supervised because labelled answers are provided during training, and binary because there are two possible classes.",
    tryThis: "Look only at the XOR dots and identify the two colours. Ignore the controls for now.",
    bonusFact: "Binary describes the two output classes, not the input values. Each dot still contains two continuous measurements: x and y.",
    target: "dataset",
    quiz: {
      question: "What is the network learning?",
      choices: ["The page colour", "Which class a dot belongs to", "How many dots exist"],
      answer: 1,
      explanation: "Correct. It uses a dot's position to predict one of two class labels.",
    },
  },
  {
    id: "examples",
    chapter: 1,
    chapterTitle: "Understanding predictions",
    eyebrow: "Training examples",
    title: "Every dot is one example",
    body: "A dot is a tiny data record containing a horizontal position, a vertical position and a correct colour.",
    why: "Displaying the records as dots lets you judge the learned rule with your eyes. Larger projects may give each example millions of input values instead.",
    technical: "Each sample has the form (x, y, label), where x and y are numerical features and label is either 0 or 1.",
    tryThis: "Notice the four alternating regions. Nearby position alone is not always enough to determine the label.",
    bonusFact: "Spreadsheet rows, images and sounds can all become training examples once they are represented numerically.",
    target: "dataset",
  },
  {
    id: "features",
    chapter: 1,
    chapterTitle: "Understanding predictions",
    eyebrow: "Inputs and features",
    title: "Features are the clues available to the network",
    body: "The network can inspect only two clues: a dot's x position and y position. These numbers are called features or inputs.",
    why: "A model cannot use information it was never given. Choosing appropriate inputs is therefore part of designing a machine-learning system.",
    technical: "The input vector is x = [x₁, x₂]. The network learns a function mapping this two-dimensional vector to one output probability.",
    tryThis: "Read the two axes as measurements supplied to the model, not as answers.",
    bonusFact: "Feature engineering creates useful measurements from raw data. Neural networks can learn internal features, but their original inputs still set the information limit.",
    target: "dataset-controls",
  },
  {
    id: "labels",
    chapter: 1,
    chapterTitle: "Understanding predictions",
    eyebrow: "Correct answers",
    title: "Labels tell the network what it should predict",
    body: "Cream dots use label 0 and teal dots use label 1. During training, the model compares its probability with this known answer.",
    why: "The label teaches this supervised model which predictions count as correct. It is used during training but is not supplied when predicting a new example.",
    technical: "The target y belongs to {0,1}. It is passed to the loss function, not to the network's input layer.",
    tryThis: "Compare the outlined dot colours with the background prediction colour.",
    bonusFact: "Inconsistent labels create a ceiling on model quality because a model can faithfully learn mistakes in its training data.",
    target: "decision-map",
  },
  {
    id: "initial-guesses",
    chapter: 1,
    chapterTitle: "Understanding predictions",
    eyebrow: "Before learning",
    title: "An untrained network begins with uninformed guesses",
    body: "Its connection weights start from seeded random values, so the first decision map usually does not match the XOR labels.",
    why: "The network must begin somewhere. Small random differences also stop all neurons in a layer from learning identical behaviour.",
    technical: "The lab uses deterministic Xavier-style initialisation. The seed recreates the same starting weights while scaling keeps early signals manageable.",
    tryThis: "Find places where a dot sits on the opposite background colour. Those are current mistakes.",
    bonusFact: "Initialising every weight to zero would leave same-layer neurons symmetrical, preventing them from developing different roles.",
    target: "decision-map",
  },
  {
    id: "probabilities",
    chapter: 1,
    chapterTitle: "Understanding predictions",
    eyebrow: "Confidence",
    title: "The output is a probability, not merely a colour",
    body: "For every position, the model produces a number between 0 and 1. Values near 0 favour cream; values near 1 favour teal.",
    why: "A probability preserves more information than a class name. Predictions of 51% and 99% select the same class but express different confidence.",
    technical: "The output uses σ(z)=1/(1+e⁻ᶻ), a sigmoid function that converts an unrestricted weighted sum into a value between 0 and 1.",
    tryThis: "Click anywhere on Experiment A's decision map and read the probability beneath it.",
    bonusFact: "Probability-like output can still be poorly calibrated. If predictions labelled 80% are correct only half the time, the model is overconfident.",
    target: "decision-map",
    action: { type: "probe", prompt: "Inspect one prediction by clicking the decision map." },
  },
  {
    id: "threshold",
    chapter: 1,
    chapterTitle: "Understanding predictions",
    eyebrow: "From probability to class",
    title: "A threshold converts probability into a decision",
    body: "The lab classifies values below 0.5 as Class 0 and values of 0.5 or above as Class 1.",
    why: "The threshold is a separate decision rule. It can be moved in real systems when different kinds of mistakes have different costs.",
    technical: "The predicted class is ŷ=1 when p≥0.5 and ŷ=0 otherwise. Accuracy uses ŷ, while loss uses the complete probability p.",
    tryThis: "Compare a pale boundary region with a strongly cream or teal region.",
    bonusFact: "A screening system may use a lower threshold to catch more possible cases while accepting more false alarms.",
    target: "metrics",
    quiz: {
      question: "Which predictions select Class 1 here?",
      choices: ["Only exactly 1.0", "Probabilities of 0.5 or higher", "Probabilities below 0.5"],
      answer: 1,
      explanation: "Correct. The 0.5 threshold converts probability into a binary decision.",
    },
  },
];

const chapterTwoSegments: CourseSegment[] = [
  {
    id: "mistakes",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Learning begins with error",
    title: "The network needs a way to measure mistakes",
    body: "A wrong prediction is useful during training because it shows that some weights should change. The network compares every probability with its label.",
    why: "Knowing only that an answer is wrong is not enough. Training needs a numerical measurement that indicates whether a proposed change improves the model.",
    technical: "For every sample, prediction p is compared with target y. The sample losses are averaged before gradients are calculated.",
    tryThis: "Find a dot sitting on a strongly opposite-coloured background. That is a confident mistake.",
    bonusFact: "Machine learning separates the rule for measuring error from the rule for updating parameters: the loss function measures; the optimiser updates.",
    target: "decision-map",
  },
  {
    id: "loss",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Wrongness score",
    title: "Loss summarises how poor the predictions are",
    body: "Loss is a confidence-sensitive wrongness score. Lower is better. It can improve even before accuracy crosses enough thresholds to change.",
    why: "A smooth score gives training a direction to move when the number of correct class decisions has not changed yet.",
    technical: "The displayed value is mean binary cross-entropy over the training set. It is differentiable with respect to the network parameters.",
    tryThis: "Read the initial loss and remember it for comparison after training.",
    bonusFact: "Loss values are meaningful mainly under the same task and definition. A value from another dataset or objective may not be directly comparable.",
    target: "metrics",
  },
  {
    id: "confident-errors",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Confidence matters",
    title: "Confident mistakes receive a larger penalty",
    body: "Predicting 51% teal for a cream dot is wrong, but predicting 99% teal is much worse. Cross-entropy represents that difference.",
    why: "The model is encouraged not only to cross the correct threshold, but also to avoid unjustified confidence.",
    technical: "Binary cross-entropy is −[y log(p)+(1−y)log(1−p)]. A completely wrong, highly confident prediction receives a sharply increasing penalty.",
    tryThis: "Contrast what loss and accuracy would report for those two wrong predictions.",
    bonusFact: "Logarithms make the penalty especially sensitive near completely wrong, completely confident predictions while remaining convenient for optimisation.",
    target: "metrics",
    quiz: {
      question: "Which wrong prediction should receive larger loss?",
      choices: ["51% teal for cream", "99% teal for cream", "They are identical"],
      answer: 1,
      explanation: "Correct. Cross-entropy strongly penalises being confidently wrong.",
    },
  },
  {
    id: "weights",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Learned connections",
    title: "Weights control the influence of each connection",
    body: "Every connection has a numerical weight. Positive, negative and near-zero weights make incoming signals matter in different ways.",
    why: "Training does not write explicit rules such as ‘top-left means cream’. It adjusts many weights until their combined calculation creates a useful boundary.",
    technical: "A neuron first calculates z=w·x+b, a weighted sum plus bias. An activation function then transforms z into the neuron's output.",
    tryThis: "Inspect the network graph and trace the adjustable connections between layers.",
    bonusFact: "Biases are adjustable offsets. They let activations and learned boundaries shift away from the origin.",
    target: "network-graph",
  },
  {
    id: "gradients",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Direction of improvement",
    title: "A gradient estimates how each parameter affects loss",
    body: "Imagine nudging one weight and checking whether loss rises or falls. A gradient provides that directional information for every weight efficiently.",
    why: "Trying random changes across many parameters would be wasteful. Gradients point towards local changes expected to reduce current loss.",
    technical: "The derivative ∂L/∂w measures local slope. Gradient descent applies w←w−η∂L/∂w.",
    tryThis: "Think of each connection as having a small arrow showing how it should move to reduce loss.",
    bonusFact: "A gradient describes local slope, not a guaranteed route to the single best solution. Neural-network loss surfaces can have flat regions and many useful minima.",
    target: "network-graph",
  },
  {
    id: "backpropagation",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Sharing responsibility",
    title: "Backpropagation carries error information backwards",
    body: "The prediction is calculated from input to output. Backpropagation then works backwards to calculate each connection's contribution to the error.",
    why: "A hidden weight affects loss only through later neurons. Backpropagation efficiently accounts for that chain of dependencies.",
    technical: "Backpropagation repeatedly applies the chain rule through the computational graph. It calculates gradients; gradient descent uses them.",
    tryThis: "Follow the graph forward to the output, then imagine error information moving back through the same layers.",
    bonusFact: "Backpropagation is not the network explaining its reasoning. It is an efficient derivative-calculation procedure used during training.",
    target: "training",
  },
  {
    id: "epochs",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Repeat and improve",
    title: "One epoch processes the training set once",
    body: "Learning normally requires many passes over the examples. Each pass gives the network another opportunity to reduce error.",
    why: "A single update rarely discovers a useful boundary. Short groups of epochs make the gradual change easier to follow.",
    technical: "This lab uses full-batch gradient descent, so one epoch calculates an update from every training example. Many larger systems use mini-batches.",
    tryThis: "Press +25 epochs in Experiment A once and watch the boundary and metrics change.",
    bonusFact: "An epoch is not a universal amount of computation. Its cost depends on dataset size, model size and batching.",
    target: "training",
    action: { type: "train-step", prompt: "Train Experiment A for 25 epochs." },
  },
  {
    id: "learning-rate",
    chapter: 2,
    chapterTitle: "How learning works",
    eyebrow: "Size of each update",
    title: "The learning rate controls how boldly weights move",
    body: "A small rate makes cautious changes. A larger rate may learn faster, but one that is too large can overshoot useful values and destabilise loss.",
    why: "Training speed and stability are linked. There is no single rate that is best for every model and dataset.",
    technical: "The η in w←w−η∇L is the learning rate. It scales each update without changing the gradient direction.",
    tryThis: "Choose a different learning rate in Experiment A. The model resets so the new run begins fairly.",
    bonusFact: "Many optimisers adapt effective step sizes, and schedules deliberately reduce the learning rate as training progresses.",
    target: "architecture",
    action: { type: "change-learning-rate", prompt: "Choose a different learning rate in Experiment A." },
  },
];

const chapterThreeSegments: CourseSegment[] = [
  {
    id: "network-structure",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "Layers",
    title: "Inputs flow through hidden layers to an output",
    body: "The compact architecture is written 2→3→1: two input values, three hidden neurons and one output probability.",
    why: "Layers transform simple inputs into useful internal representations before the final decision.",
    technical: "A dense layer connects every output neuron to every value in the previous layer. The final layer contains one sigmoid output for binary classification.",
    tryThis: "Read the graph from left to right and match each column with 2→3→1.",
    bonusFact: "Layer sizes are hyperparameters selected outside training, whereas weights and biases are parameters learned during training.",
    target: "network-graph",
  },
  {
    id: "individual-neuron",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "One neuron",
    title: "A neuron performs a small mathematical transformation",
    body: "It combines weighted inputs, adds a bias and applies an activation. A network becomes powerful by composing many small operations.",
    why: "A single neuron produces a simple boundary. Hidden layers combine neurons so the overall model can represent more complicated shapes.",
    technical: "For input x, weights w and bias b, a hidden neuron outputs a=φ(w·x+b).",
    tryThis: "Pick one hidden node and trace its incoming and outgoing connections.",
    bonusFact: "Artificial neurons are simplified mathematical units. Their name is historically inspired by biology, but they are not realistic simulations of brain cells.",
    target: "network-graph",
    quiz: {
      question: "What does training learn in this network?",
      choices: ["Only screen colours", "Weights and biases", "The number of inputs"],
      answer: 1,
      explanation: "Correct. Training adjusts weights and biases; the architecture is chosen separately.",
    },
  },
  {
    id: "xor-nonlinearity",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "Why XOR matters",
    title: "XOR cannot be solved with one straight boundary",
    body: "Its alternating quadrants defeat a purely linear classifier. A useful network must combine partial boundaries into a non-linear region.",
    why: "This tiny problem demonstrates why hidden layers and non-linear activations matter. Stacking linear operations alone would remain linear.",
    technical: "XOR is not linearly separable in its original two-dimensional space. A hidden layer can transform the representation before the output separates it.",
    tryThis: "Look at the two disconnected teal regions and try to imagine one line separating both from cream.",
    bonusFact: "The historical XOR debate helped motivate multi-layer networks and effective training methods, although the broader history is more complex than one breakthrough.",
    target: "decision-map",
  },
  {
    id: "activations",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "Non-linear transformations",
    title: "Activation functions change how neurons respond",
    body: "Tanh, ReLU and sigmoid transform weighted sums differently. Their shapes influence representations and gradient flow.",
    why: "Without non-linearity between dense layers, adding more layers would not produce genuinely curved or disconnected regions.",
    technical: "Tanh outputs −1 to 1, sigmoid outputs 0 to 1, and ReLU outputs max(0,z). Their derivatives also differ during backpropagation.",
    tryThis: "Change Experiment A's activation. The network resets because the same weights would now behave differently.",
    bonusFact: "ReLU became common partly because its positive-side gradient does not shrink as sigmoid and tanh gradients can in saturated regions.",
    target: "architecture",
    action: { type: "change-activation", prompt: "Choose a different activation in Experiment A." },
  },
  {
    id: "width-depth",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "Capacity",
    title: "Width and depth change what the network can represent",
    body: "Wider layers contain more neurons; deeper networks contain more transformations. Both increase capacity and usually computation.",
    why: "More capacity can help with complicated patterns, but it is not automatically better. Small networks may learn faster and remain easier to understand.",
    technical: "Changing architecture changes the number and arrangement of trainable parameters. Compare performance, stability and complexity together.",
    tryThis: "Select Balanced or Deep in Experiment A and inspect the new graph.",
    bonusFact: "A model may have enough theoretical capacity yet train poorly. Expressive power and successful optimisation are different questions.",
    target: "architecture",
    action: { type: "change-architecture", prompt: "Choose a different architecture in Experiment A." },
  },
  {
    id: "underfitting",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "Too simple",
    title: "Underfitting means the model misses the pattern",
    body: "An underfitting model remains systematically wrong after reasonable training because its representation or training setup is too limited.",
    why: "Low performance is not always solved by more epochs. Architecture, features or optimisation may be preventing useful learning.",
    technical: "Underfitting is associated with high bias: the model's assumptions are too restrictive for the data-generating pattern.",
    tryThis: "Compare a simple early boundary with the two disconnected regions required by XOR.",
    bonusFact: "A large model can underfit too if training stops early, the learning rate is ineffective or essential inputs are missing.",
    target: "decision-map",
  },
  {
    id: "train-test",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "Unseen examples",
    title: "Test data checks whether learning generalises",
    body: "Train correct measures examples used for weight updates. Test correct measures a separate clean set that the optimiser never receives.",
    why: "A useful model learns a pattern beyond memorising its training examples. A persistent gap between training and test performance may reveal trouble.",
    technical: "The lab creates a deterministic hold-out set from a different seed. Test samples are evaluated but never passed to trainEpoch.",
    tryThis: "Compare Train correct and Test correct. Small differences are normal; a large persistent gap deserves investigation.",
    bonusFact: "Repeatedly changing a model based on test results can indirectly overfit the test set, so real projects often preserve a final untouched set.",
    target: "metrics",
    quiz: {
      question: "Which examples update the weights?",
      choices: ["Training examples only", "Test examples only", "Both sets"],
      answer: 0,
      explanation: "Correct. The separate test set is reserved for measuring generalisation.",
    },
  },
  {
    id: "noise-overfitting",
    chapter: 3,
    chapterTitle: "Designing the network",
    eyebrow: "Memorising accidents",
    title: "Noise can expose overfitting",
    body: "Label noise flips some training answers while leaving the test set clean. A flexible model may begin fitting those misleading examples.",
    why: "If training performance improves while clean test performance stalls or falls, the model may be memorising peculiarities instead of the rule.",
    technical: "Overfitting is associated with high variance. Noise, excessive capacity and prolonged training can contribute to it.",
    tryThis: "Set label noise to 15%. Training labels change while the unseen test set remains clean.",
    bonusFact: "Regularisation, representative data, early stopping and simpler models can all help reduce overfitting.",
    target: "dataset-controls",
    action: { type: "change-noise", prompt: "Set label noise to a non-zero value." },
  },
];

const chapterFourSegments: CourseSegment[] = [
  {
    id: "seeds",
    chapter: 4,
    chapterTitle: "Running meaningful experiments",
    eyebrow: "Reproducibility",
    title: "A seed recreates the same random choices",
    body: "The seed controls generated examples and starting weights. Reusing it lets another run begin from the same conditions.",
    why: "Reproducibility makes comparison and debugging possible. It stops apparent improvement from being caused merely by easier random examples.",
    technical: "A deterministic pseudo-random generator turns a seed into a repeatable sequence. One seed is repeatable, not automatically representative.",
    tryThis: "Change the seed and watch the examples and initial boundaries reset.",
    bonusFact: "Good results are eventually checked across several seeds. Reproducing one run confirms procedure; repeated runs reveal sensitivity to randomness.",
    target: "dataset-controls",
    action: { type: "change-seed", prompt: "Enter a different seed." },
  },
  {
    id: "controlled-comparison",
    chapter: 4,
    chapterTitle: "Running meaningful experiments",
    eyebrow: "One variable at a time",
    title: "A fair comparison changes one deliberate factor",
    body: "Experiment A is the baseline and B is the challenger. Both receive the same data so you can isolate a configuration change.",
    why: "If architecture, activation, learning rate and data all change together, you cannot tell which difference caused the outcome.",
    technical: "The changed configuration is the independent variable. Loss, accuracy and boundary shape are dependent observations.",
    tryThis: "Before editing B, identify which A settings should remain fixed.",
    bonusFact: "Reproducible ML experiments also record preprocessing and software versions because they can affect results.",
    target: "comparison",
    quiz: {
      question: "For a fair architecture comparison, what else stays the same?",
      choices: ["Data, seed, activation and rate", "Nothing", "Everything including architecture"],
      answer: 0,
      explanation: "Correct. Hold the other conditions constant so architecture is the main changed variable.",
    },
  },
  {
    id: "architecture-duel",
    chapter: 4,
    chapterTitle: "Running meaningful experiments",
    eyebrow: "Architecture experiment",
    title: "Compare a compact baseline with a larger challenger",
    body: "A larger network may learn a more complicated boundary, but may also add unnecessary parameters or react more strongly to noise.",
    why: "The goal is not to award victory to the biggest model. It is to find sufficient capacity while considering stability and generalisation.",
    technical: "Model selection balances empirical performance with complexity. On small datasets, a simpler hypothesis can generalise more reliably.",
    tryThis: "Change only Experiment B's architecture.",
    bonusFact: "Parameter count grows with every connection, so adding neurons to adjacent dense layers can increase parameters quickly.",
    target: "comparison",
    action: { type: "change-architecture-b", prompt: "Change Experiment B's architecture." },
  },
  {
    id: "activation-showdown",
    chapter: 4,
    chapterTitle: "Running meaningful experiments",
    eyebrow: "Activation experiment",
    title: "Hold architecture constant and compare activations",
    body: "Activations can produce different learning trajectories even when data, architecture and learning rate match.",
    why: "The full curve matters. Models may reach similar final accuracy while differing in speed and stability.",
    technical: "Activations affect both forward representations and backward derivatives. Saturating functions can produce very small gradients.",
    tryThis: "Change Experiment B's activation and keep the other settings aligned with A.",
    bonusFact: "No activation wins every task. ReLU, tanh and sigmoid impose different behaviours and optimisation trade-offs.",
    target: "comparison",
    action: { type: "change-activation-b", prompt: "Change Experiment B's activation." },
  },
  {
    id: "reading-evidence",
    chapter: 4,
    chapterTitle: "Running meaningful experiments",
    eyebrow: "Interpret the whole run",
    title: "Read loss, accuracy and boundaries together",
    body: "Loss shows confidence-sensitive improvement, accuracy counts decisions, test accuracy checks generalisation and the map reveals shape.",
    why: "A high accuracy number can conceal an unreasonable boundary or confident errors. Visual and numerical evidence should support one another.",
    technical: "A falling training loss with worsening test accuracy is a classic warning sign of overfitting.",
    tryThis: "Train both experiments, then compare their loss, train accuracy, test accuracy and decision maps.",
    bonusFact: "Strong evaluation uses metrics chosen for the real cost of errors, not whichever single number looks largest.",
    target: "loss-chart",
    action: { type: "train-both", prompt: "Train both Experiment A and Experiment B at least once." },
  },
  {
    id: "playground",
    chapter: 4,
    chapterTitle: "Running meaningful experiments",
    eyebrow: "The full lab",
    title: "You now have the complete experimental loop",
    body: "Choose data, introduce noise, configure networks, inspect predictions, train in steps, compare clean test performance and explore alternate representations.",
    why: "The lab is most useful when you form a question before moving controls, predict what will happen, run a controlled test and explain the evidence.",
    technical: "Models, backpropagation, metrics and seeded data all run locally in TypeScript. Node physics changes exploration, not training.",
    tryThis: "Enable Node physics. Finish the guide, then choose a challenge or design your own experiment.",
    bonusFact: "Physics changes only graph layout; dragging a node never changes a weight.",
    target: "modes",
    action: { type: "enable-mode", prompt: "Enable Node physics." },
  },
];

const chapterFiveSegments: CourseSegment[] = [
  {
    id: "workbench-map",
    chapter: 5,
    chapterTitle: "Building a network by hand",
    eyebrow: "The open workbench",
    title: "The bottom canvas exposes the individual parts",
    body: "The comparison lab uses dense layers, where neighbouring layers are automatically fully connected. The Neural Workbench lets you place neurons and decide which individual connections should exist.",
    why: "A freeform graph makes the architecture concrete. You can see that a network is not a mysterious block, but a collection of calculations joined by weighted paths.",
    technical: "The workbench stores a directed graph of input, hidden and output nodes. Every connection has its own weight and every non-input node has a bias.",
    tryThis: "Scroll around the workbench and identify the two inputs, three hidden neurons, output neuron and coloured weighted connections.",
    bonusFact: "Dense networks are convenient, but sparse networks can use far fewer connections. Sparsity is an active topic in efficient machine learning.",
    target: "workbench",
  },
  {
    id: "workbench-editing",
    chapter: 5,
    chapterTitle: "Building a network by hand",
    eyebrow: "Nodes and connections",
    title: "You can change both the diagram and the calculation",
    body: "Add inputs or hidden neurons, drag them into place, and use the connection tool to wire a source to a destination. Selecting a neuron or edge opens its editable properties.",
    why: "Unlike the physics mode above, edits here are functional. Changing a weight, bias, activation or connection changes the result of the graph.",
    technical: "Positive weights are shown in petrol and negative weights in rust. The editor rejects duplicate links, links into inputs and links that would create a cycle.",
    tryThis: "Add one hidden neuron, connect it to the output and adjust the new connection's weight.",
    bonusFact: "A negative weight does not mean a bad connection. It means the source activation pushes the target's weighted sum downward rather than upward.",
    target: "workbench",
  },
  {
    id: "workbench-training",
    chapter: 5,
    chapterTitle: "Building a network by hand",
    eyebrow: "Run and train",
    title: "Manual pulses and dataset training answer different questions",
    body: "The input sliders show one forward pass. Test dataset measures the whole shared dataset, while Train 40 steps updates the graph's weights and biases using those labelled examples.",
    why: "Inspecting one input helps explain a calculation. Dataset metrics help judge whether the same graph has learned a pattern that works across many examples.",
    technical: "The workbench performs topological forward evaluation and reverse-mode backpropagation over its editable directed acyclic graph.",
    tryThis: "Move the two input sliders, run a pulse, then train 40 steps and compare the loss and accuracy readout.",
    bonusFact: "The graph's visual positions do not determine its computation. Connections determine the data flow; dragging a neuron changes only the diagram.",
    target: "workbench",
  },
];

const courseSegments: CourseSegment[] = [
  ...chapterOneSegments,
  ...chapterTwoSegments,
  ...chapterThreeSegments,
  ...chapterFourSegments,
  ...chapterFiveSegments,
];

function applyLabelNoise(
  samples: Sample[],
  percentage: number,
  seed: number,
): Sample[] {
  if (percentage <= 0) return samples;
  const random = createRng(seed + 7013);
  return samples.map((sample) =>
    random() < percentage / 100
      ? { ...sample, label: (sample.label === 1 ? 0 : 1) as 0 | 1 }
      : sample,
  );
}

function DecisionMap({
  network,
  activation,
  samples,
  revision,
  probe,
  onProbe,
}: {
  network: NeuralNetwork;
  activation: ActivationName;
  samples: Sample[];
  revision: number;
  probe: Probe | null;
  onProbe: (probe: Probe) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const pointFromEvent = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      const rectangle = event.currentTarget.getBoundingClientRect();
      const x = ((event.clientX - rectangle.left) / rectangle.width) * 2 - 1;
      const y = 1 - ((event.clientY - rectangle.top) / rectangle.height) * 2;
      return { x, y, probability: forward(network, [x, y], activation).output };
    },
    [activation, network],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const { width, height } = canvas;
    const columns = 52;
    const rows = 38;
    const cellWidth = width / columns;
    const cellHeight = height / rows;

    context.clearRect(0, 0, width, height);
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const x = ((column + 0.5) / columns) * 2 - 1;
        const y = 1 - ((row + 0.5) / rows) * 2;
        const probability = forward(network, [x, y], activation).output;
        const red = Math.round(239 - probability * 197);
        const green = Math.round(225 - probability * 113);
        const blue = Math.round(207 - probability * 100);
        context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        context.fillRect(
          column * cellWidth,
          row * cellHeight,
          Math.ceil(cellWidth + 0.5),
          Math.ceil(cellHeight + 0.5),
        );
      }
    }

    context.strokeStyle = "rgba(24, 35, 33, 0.2)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(width / 2, 0);
    context.lineTo(width / 2, height);
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
    context.stroke();

    samples.forEach((sample) => {
      const px = ((sample.x + 1) / 2) * width;
      const py = ((1 - sample.y) / 2) * height;
      context.beginPath();
      context.arc(px, py, 4.2, 0, Math.PI * 2);
      context.fillStyle = sample.label === 1 ? "#0f5f58" : "#fff8eb";
      context.fill();
      context.strokeStyle = sample.label === 1 ? "#e1f0eb" : "#3c2f25";
      context.lineWidth = 1.5;
      context.stroke();
    });

    if (probe) {
      const px = ((probe.x + 1) / 2) * width;
      const py = ((1 - probe.y) / 2) * height;
      context.strokeStyle = "#111827";
      context.lineWidth = 2;
      context.beginPath();
      context.arc(px, py, 8, 0, Math.PI * 2);
      context.moveTo(px - 12, py);
      context.lineTo(px + 12, py);
      context.moveTo(px, py - 12);
      context.lineTo(px, py + 12);
      context.stroke();
    }
  }, [activation, network, probe, revision, samples]);

  return (
    <canvas
      ref={canvasRef}
      width={520}
      height={350}
      className="decision-canvas"
      aria-label="Interactive neural-network decision map. Click to inspect a prediction."
      onClick={(event) => onProbe(pointFromEvent(event))}
    />
  );
}

function LossChart({ history }: { history: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const { width, height } = canvas;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#f7f4ed";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "rgba(26, 34, 56, 0.12)";
    [0.25, 0.5, 0.75].forEach((fraction) => {
      context.beginPath();
      context.moveTo(0, height * fraction);
      context.lineTo(width, height * fraction);
      context.stroke();
    });
    if (history.length < 2) return;
    const visible = history.slice(-240);
    const maximum = Math.max(0.72, ...visible);
    context.beginPath();
    visible.forEach((loss, index) => {
      const x = (index / Math.max(1, visible.length - 1)) * width;
      const y = height - Math.min(1, loss / maximum) * (height - 8) - 4;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = "#176b65";
    context.lineWidth = 3;
    context.lineJoin = "round";
    context.stroke();
  }, [history]);

  return (
    <canvas
      ref={canvasRef}
      width={520}
      height={100}
      className="loss-canvas"
      aria-label="Loss curve for the current training run"
    />
  );
}

function NetworkGraph({
  architecture,
  physics,
}: {
  architecture: number[];
  physics: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const layerSizes = [2, ...architecture, 1];
    const nodes = layerSizes.flatMap((size, layer) =>
      Array.from({ length: size }, (_, index) => {
        const anchorX =
          36 + (layer / (layerSizes.length - 1)) * (canvas.width - 72);
        const anchorY =
          canvas.height / 2 +
          (index - (size - 1) / 2) * Math.min(27, 126 / size);
        return {
          layer,
          anchorX,
          anchorY,
          x: anchorX,
          y: anchorY,
          vx: 0,
          vy: 0,
        };
      }),
    );
    let animationFrame = 0;
    let draggedNode: (typeof nodes)[number] | null = null;

    const draw = () => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#172522";
      context.fillRect(0, 0, canvas.width, canvas.height);
      for (let layer = 0; layer < layerSizes.length - 1; layer += 1) {
        const leftNodes = nodes.filter((node) => node.layer === layer);
        const rightNodes = nodes.filter((node) => node.layer === layer + 1);
        leftNodes.forEach((left) =>
          rightNodes.forEach((right) => {
            context.beginPath();
            context.moveTo(left.x, left.y);
            context.lineTo(right.x, right.y);
            context.strokeStyle = "rgba(194, 226, 215, 0.22)";
            context.lineWidth = 1;
            context.stroke();
          }),
        );
      }
      nodes.forEach((node) => {
        context.beginPath();
        context.arc(node.x, node.y, 7, 0, Math.PI * 2);
        context.fillStyle =
          node.layer === 0
            ? "#d4a13a"
            : node.layer === layerSizes.length - 1
              ? "#d97851"
              : "#61b8ad";
        context.fill();
        context.strokeStyle = "rgba(255, 255, 255, 0.75)";
        context.lineWidth = 1;
        context.stroke();
      });
    };

    const animate = () => {
      if (!physics) {
        draw();
        return;
      }
      nodes.forEach((node, nodeIndex) => {
        if (node === draggedNode) return;
        node.vx += (node.anchorX - node.x) * 0.006;
        node.vy += (node.anchorY - node.y) * 0.006;
        nodes.forEach((other, otherIndex) => {
          if (nodeIndex === otherIndex || node.layer !== other.layer) return;
          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const distanceSquared = Math.max(35, dx * dx + dy * dy);
          node.vx += (dx / distanceSquared) * 0.16;
          node.vy += (dy / distanceSquared) * 0.16;
        });
        node.vx *= 0.92;
        node.vy *= 0.92;
        node.x += node.vx;
        node.y += node.vy;
      });
      draw();
      animationFrame = requestAnimationFrame(animate);
    };

    const canvasPoint = (event: PointerEvent) => {
      const rectangle = canvas.getBoundingClientRect();
      return {
        x: ((event.clientX - rectangle.left) / rectangle.width) * canvas.width,
        y: ((event.clientY - rectangle.top) / rectangle.height) * canvas.height,
      };
    };
    const pointerDown = (event: PointerEvent) => {
      const point = canvasPoint(event);
      draggedNode =
        nodes.find((node) => Math.hypot(node.x - point.x, node.y - point.y) < 14) ??
        null;
      if (draggedNode) canvas.setPointerCapture(event.pointerId);
    };
    const pointerMove = (event: PointerEvent) => {
      if (!draggedNode) return;
      const point = canvasPoint(event);
      draggedNode.x = point.x;
      draggedNode.y = point.y;
      draggedNode.vx = 0;
      draggedNode.vy = 0;
      if (!physics) draw();
    };
    const pointerUp = () => {
      draggedNode = null;
    };

    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);
    animate();
    return () => {
      cancelAnimationFrame(animationFrame);
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("pointercancel", pointerUp);
    };
  }, [architecture, physics]);

  return (
    <canvas
      ref={canvasRef}
      width={520}
      height={170}
      className="network-canvas"
      aria-label="Network node graph. Drag nodes to inspect the physics interaction."
    />
  );
}

type TutorialVisibility = {
  controls: boolean;
  metrics: boolean;
  training: boolean;
  decisionMap: boolean;
  lossChart: boolean;
  networkGraph: boolean;
};

function Experiment({
  label,
  accent,
  initialConfig,
  dataset,
  testDataset,
  seed,
  physics,
  guideTarget,
  visibility,
  onTutorialAction,
}: {
  label: string;
  accent: "violet" | "blue";
  initialConfig: ExperimentConfig;
  dataset: Sample[];
  testDataset: Sample[];
  seed: number;
  physics: boolean;
  guideTarget: string | null;
  visibility: TutorialVisibility;
  onTutorialAction: (event: TutorialEvent) => void;
}) {
  const seedOffset = label === "A" ? 101 : 907;
  const [config, setConfig] = useState(initialConfig);
  const [network, setNetwork] = useState(() =>
    createNetwork(
      architectureFromValue(initialConfig.architecture),
      seed + seedOffset,
    ),
  );
  const [metrics, setMetrics] = useState<Metrics>(() =>
    evaluate(network, dataset, initialConfig.activation),
  );
  const [testMetrics, setTestMetrics] = useState<Metrics>(() =>
    evaluate(network, testDataset, initialConfig.activation),
  );
  const [history, setHistory] = useState<number[]>(() => [metrics.loss]);
  const [epoch, setEpoch] = useState(0);
  const epochRef = useRef(0);
  const [revision, setRevision] = useState(0);
  const [running, setRunning] = useState(false);
  const [probe, setProbe] = useState<Probe | null>(null);
  const hasMounted = useRef(false);

  const reset = useCallback(
    (nextConfig = config) => {
      setRunning(false);
      const network = createNetwork(
        architectureFromValue(nextConfig.architecture),
        seed + seedOffset,
      );
      setNetwork(network);
      const nextMetrics = evaluate(network, dataset, nextConfig.activation);
      const nextTestMetrics = evaluate(
        network,
        testDataset,
        nextConfig.activation,
      );
      epochRef.current = 0;
      setEpoch(0);
      setMetrics(nextMetrics);
      setTestMetrics(nextTestMetrics);
      setHistory([nextMetrics.loss]);
      setProbe(null);
      setRevision((value) => value + 1);
    },
    [config, dataset, seed, seedOffset, testDataset],
  );

  useEffect(() => {
    if (!hasMounted.current) {
      hasMounted.current = true;
      return;
    }
    reset(config);
    // Reset only when the shared data changes; configuration changes reset directly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, seed, testDataset]);

  useEffect(() => {
    if (!running) return;
    let frame = 0;
    const trainFrame = () => {
      let nextMetrics = evaluate(
        network,
        dataset,
        config.activation,
      );
      const newLosses: number[] = [];
      for (let step = 0; step < 20; step += 1) {
        nextMetrics = trainEpoch(
          network,
          dataset,
          config.activation,
          config.learningRate,
        );
        newLosses.push(nextMetrics.loss);
        epochRef.current += 1;
      }
      setEpoch(epochRef.current);
      setMetrics(nextMetrics);
      setTestMetrics(
        evaluate(network, testDataset, config.activation),
      );
      setHistory((values) => [...values, ...newLosses].slice(-400));
      setRevision((value) => value + 1);
      if (epochRef.current >= 600) {
        setRunning(false);
        return;
      }
      frame = requestAnimationFrame(trainFrame);
    };
    frame = requestAnimationFrame(trainFrame);
    return () => cancelAnimationFrame(frame);
  }, [config.activation, config.learningRate, dataset, network, running, testDataset]);

  const updateConfig = <Key extends keyof ExperimentConfig>(
    key: Key,
    value: ExperimentConfig[Key],
  ) => {
    const next = { ...config, [key]: value };
    setConfig(next);
    reset(next);
    const names: Record<keyof ExperimentConfig, string> = {
      architecture: "architecture",
      activation: "activation",
      learningRate: "learning rate",
    };
    onTutorialAction({
      type:
        key === "architecture"
          ? "change-architecture"
          : key === "activation"
            ? "change-activation"
            : "change-learning-rate",
      experiment: label as "A" | "B",
      feedback: `Experiment ${label}'s ${names[key]} changed to ${String(value)}. The model was reset for a fair new run.`,
    });
  };

  const stepTraining = () => {
    setRunning(false);
    let nextMetrics = metrics;
    const newLosses: number[] = [];
    for (let step = 0; step < 25; step += 1) {
      nextMetrics = trainEpoch(
        network,
        dataset,
        config.activation,
        config.learningRate,
      );
      newLosses.push(nextMetrics.loss);
      epochRef.current += 1;
    }
    setEpoch(epochRef.current);
    setMetrics(nextMetrics);
    const nextTestMetrics = evaluate(
      network,
      testDataset,
      config.activation,
    );
    setTestMetrics(nextTestMetrics);
    setHistory((values) => [...values, ...newLosses].slice(-400));
    setRevision((value) => value + 1);
    onTutorialAction({
      type: "train-step",
      experiment: label as "A" | "B",
      feedback: `Experiment ${label} reached epoch ${epochRef.current}. Loss is ${nextMetrics.loss.toFixed(3)}, train accuracy is ${Math.round(nextMetrics.accuracy * 100)}%, and test accuracy is ${Math.round(nextTestMetrics.accuracy * 100)}%.`,
    });
  };

  const toggleTraining = () => {
    if (!running) {
      onTutorialAction({
        type: "train-start",
        experiment: label as "A" | "B",
        feedback: `Experiment ${label} started continuous training.`,
      });
    }
    setRunning(!running);
  };

  const architecture = architectureFromValue(config.architecture);
  const architectureIsPreset = architectureOptions.some(
    (option) => option.value === config.architecture,
  );
  const updateArchitectureLayers = (nextLayers: number[]) => {
    updateConfig(
      "architecture",
      nextLayers
        .map((size) => Math.max(1, Math.min(12, size)))
        .slice(0, 4)
        .join("-"),
    );
  };
  return (
    <article className={`experiment-card accent-${accent}`}>
      <header className="experiment-header">
        <div>
          <p className="eyebrow">Experiment {label}</p>
          <h2>{label === "A" ? "Baseline model" : "Challenger model"}</h2>
        </div>
        <span className={`status-dot ${running ? "running" : ""}`}>
          {running ? "Training" : "Ready"}
        </span>
      </header>

      {visibility.controls ? <div
        className="control-grid"
        data-guide-target="architecture"
        data-guide-active={guideTarget === "architecture" ? "true" : undefined}
      >
        <label>
          Architecture
          <select
            value={config.architecture}
            onChange={(event) => updateConfig("architecture", event.target.value)}
          >
            {!architectureIsPreset ? (
              <option value={config.architecture}>
                Custom · 2→{config.architecture.replaceAll("-", "→")}→1
              </option>
            ) : null}
            {architectureOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label} · 2→{option.value.replaceAll("-", "→")}→1
              </option>
            ))}
          </select>
        </label>
        <label>
          Activation
          <select
            value={config.activation}
            onChange={(event) =>
              updateConfig("activation", event.target.value as ActivationName)
            }
          >
            <option value="tanh">Tanh</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
          </select>
        </label>
        <label>
          Learning rate
          <select
            value={config.learningRate}
            onChange={(event) =>
              updateConfig("learningRate", Number(event.target.value))
            }
          >
            <option value={0.01}>0.01 · cautious</option>
            <option value={0.03}>0.03 · steady</option>
            <option value={0.08}>0.08 · fast</option>
          </select>
        </label>
      </div> : null}

      {visibility.controls ? (
        <div className="layer-editor" aria-label={`Edit Experiment ${label} hidden layers`}>
          <div>
            <span>Hidden layers</span>
            <small>Neighbouring layers stay fully connected.</small>
          </div>
          <div className="layer-editor-controls">
            {architecture.map((size, index) => (
              <div className="layer-stepper" key={`${index}-${architecture.length}`}>
                <button
                  type="button"
                  aria-label={`Remove one neuron from hidden layer ${index + 1}`}
                  disabled={size <= 1}
                  onClick={() =>
                    updateArchitectureLayers(
                      architecture.map((layerSize, layerIndex) =>
                        layerIndex === index ? layerSize - 1 : layerSize,
                      ),
                    )
                  }
                >
                  −
                </button>
                <span><b>{size}</b><small>L{index + 1}</small></span>
                <button
                  type="button"
                  aria-label={`Add one neuron to hidden layer ${index + 1}`}
                  disabled={size >= 12}
                  onClick={() =>
                    updateArchitectureLayers(
                      architecture.map((layerSize, layerIndex) =>
                        layerIndex === index ? layerSize + 1 : layerSize,
                      ),
                    )
                  }
                >
                  +
                </button>
              </div>
            ))}
            <button
              type="button"
              className="layer-action"
              disabled={architecture.length >= 4}
              onClick={() => updateArchitectureLayers([...architecture, 3])}
            >
              + layer
            </button>
            <button
              type="button"
              className="layer-action"
              disabled={architecture.length <= 1}
              onClick={() => updateArchitectureLayers(architecture.slice(0, -1))}
            >
              − layer
            </button>
          </div>
        </div>
      ) : null}

      {visibility.metrics ? <div
        className="metrics"
        aria-live="polite"
        data-guide-target="metrics"
        data-guide-active={guideTarget === "metrics" ? "true" : undefined}
      >
        <div><span>Epoch</span><strong>{epoch}</strong></div>
        <div><span>Loss</span><strong>{metrics.loss.toFixed(3)}</strong></div>
        <div><span>Train correct</span><strong>{Math.round(metrics.accuracy * 100)}%</strong></div>
        <div><span>Test correct</span><strong>{Math.round(testMetrics.accuracy * 100)}%</strong></div>
      </div> : null}

      {visibility.training ? <div
        className="button-row"
        data-guide-target="training"
        data-guide-active={guideTarget === "training" ? "true" : undefined}
      >
        <button className="primary-button" onClick={toggleTraining}>
          {running ? "Pause training" : epoch > 0 ? "Continue training" : "Train model"}
        </button>
        <button className="secondary-button" onClick={stepTraining}>+25 epochs</button>
        <button className="text-button" onClick={() => reset()}>Reset</button>
      </div> : null}

      {visibility.decisionMap ? <figure
        className="visual-block"
        data-guide-target="decision-map"
        data-guide-active={guideTarget === "decision-map" ? "true" : undefined}
      >
        <figcaption>
          <span>Decision map</span>
          <small>Click anywhere to inspect a prediction</small>
        </figcaption>
        <DecisionMap
          network={network}
          activation={config.activation}
          samples={dataset}
          revision={revision}
          probe={probe}
          onProbe={(nextProbe) => {
            setProbe(nextProbe);
            onTutorialAction({
              type: "probe",
              experiment: label as "A" | "B",
              feedback: `At that coordinate, Experiment ${label} predicts ${Math.round(nextProbe.probability * 100)}% probability of Class 1.`,
            });
          }}
        />
        <div className="legend" aria-hidden="true">
          <span><i className="class-zero" />Class 0</span>
          <span><i className="class-one" />Class 1</span>
          <span className="gradient-key">Prediction probability</span>
        </div>
        <p className="probe-result" aria-live="polite">
          {probe
            ? `Probe (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)}): ${Math.round(
                probe.probability * 100,
              )}% probability of Class 1`
            : "No probe selected yet."}
        </p>
      </figure> : null}

      {visibility.lossChart ? <figure
        className="visual-block compact"
        data-guide-target="loss-chart"
        data-guide-active={guideTarget === "loss-chart" ? "true" : undefined}
      >
        <figcaption>
          <span>Loss curve</span>
          <small>Last {Math.min(history.length, 240)} epochs</small>
        </figcaption>
        <LossChart history={history} />
      </figure> : null}

      {visibility.networkGraph ? <figure
        className="visual-block compact"
        data-guide-target="network-graph"
        data-guide-active={guideTarget === "network-graph" ? "true" : undefined}
      >
        <figcaption>
          <span>Network graph</span>
          <small>{physics ? "Physics active · drag a node" : "Drag a node to inspect the graph"}</small>
        </figcaption>
        <NetworkGraph architecture={architecture} physics={physics} />
      </figure> : null}
    </article>
  );
}

export default function Home() {
  const [datasetKind, setDatasetKind] = useState<DatasetKind>("moons");
  const [seed, setSeed] = useState(42);
  const [sampleCount, setSampleCount] = useState(180);
  const [noisePercent, setNoisePercent] = useState(0);
  const [experimentConfigs, setExperimentConfigs] =
    useState<[ExperimentConfig, ExperimentConfig]>(initialConfigs);
  const [physics, setPhysics] = useState(false);
  const [guideMenuOpen, setGuideMenuOpen] = useState(false);
  const [showTutorialSegments, setShowTutorialSegments] = useState(false);
  const [tutorialActive, setTutorialActive] = useState(false);
  const [tutorialIndex, setTutorialIndex] = useState(0);
  const [tutorialRun, setTutorialRun] = useState(0);
  const [seenSegments, setSeenSegments] = useState<number[]>([]);
  const [guideMessage, setGuideMessage] = useState<string | null>(null);
  const [bonusFactsEnabled, setBonusFactsEnabled] = useState(true);
  const [dismissedBonusFacts, setDismissedBonusFacts] = useState<string[]>([]);
  const [showBonusStopPrompt, setShowBonusStopPrompt] = useState(false);
  const [showWhy, setShowWhy] = useState(false);
  const [showTechnical, setShowTechnical] = useState(false);
  const [quizChoice, setQuizChoice] = useState<number | null>(null);
  const [segmentTaskComplete, setSegmentTaskComplete] = useState(true);
  const [tutorialFeedback, setTutorialFeedback] = useState<string | null>(null);
  const guideMessageTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recentBonusDismissals = useRef<number[]>([]);
  const trainedTutorialExperiments = useRef<Set<"A" | "B">>(new Set());
  const baseDataset = useMemo(
    () => generateDataset(datasetKind, seed, sampleCount),
    [datasetKind, sampleCount, seed],
  );
  const dataset = useMemo(
    () => applyLabelNoise(baseDataset, noisePercent, seed),
    [baseDataset, noisePercent, seed],
  );
  const testDataset = useMemo(
    () =>
      generateDataset(
        datasetKind,
        seed + 10007,
        Math.max(60, Math.round(sampleCount / 2)),
      ),
    [datasetKind, sampleCount, seed],
  );
  const activeTutorialSegment = courseSegments[tutorialIndex];
  const activeGuideTarget = tutorialActive ? activeTutorialSegment.target : null;
  const activeChapterSegments = courseSegments.filter(
    (segment) => segment.chapter === activeTutorialSegment.chapter,
  );
  const tutorialVisibility: TutorialVisibility = {
    controls: !tutorialActive || tutorialIndex >= 14,
    metrics: !tutorialActive || tutorialIndex >= 6,
    training: !tutorialActive || tutorialIndex >= 12,
    decisionMap: !tutorialActive || tutorialIndex >= 3,
    lossChart: !tutorialActive || tutorialIndex >= 8,
    networkGraph: !tutorialActive || tutorialIndex >= 10,
  };
  const showExperimentA = !tutorialActive || tutorialIndex >= 3;
  const showExperimentB = !tutorialActive || tutorialIndex >= 24;
  const showModes = !tutorialActive || tutorialIndex >= 28;

  const showGuideMessage = useCallback((message: string) => {
    if (guideMessageTimer.current) clearTimeout(guideMessageTimer.current);
    setGuideMessage(message);
    guideMessageTimer.current = setTimeout(() => setGuideMessage(null), 2400);
  }, []);

  const setBonusFactPreference = (enabled: boolean) => {
    setBonusFactsEnabled(enabled);
    setShowBonusStopPrompt(false);
    recentBonusDismissals.current = [];
    if (enabled) setDismissedBonusFacts([]);
    window.localStorage.setItem("nnvl-bonus-facts", enabled ? "on" : "off");
  };

  const dismissBonusFact = () => {
    setDismissedBonusFacts((facts) =>
      facts.includes(activeTutorialSegment.id)
        ? facts
        : [...facts, activeTutorialSegment.id],
    );
    const now = Date.now();
    const quickDismissals = [...recentBonusDismissals.current, now]
      .filter((time) => now - time <= 8000)
      .slice(-2);
    recentBonusDismissals.current = quickDismissals;
    if (quickDismissals.length === 2) setShowBonusStopPrompt(true);
  };

  const stopBonusFacts = () => {
    setBonusFactPreference(false);
    showGuideMessage(
      "Got it! You can turn them back on under the tutorial button.",
    );
  };

  const handleTutorialAction = (event: TutorialEvent) => {
    if (!tutorialActive || !activeTutorialSegment.action) return;
    const required = activeTutorialSegment.action.type;
    let matched = false;

    if (required === "train-both" &&
        (event.type === "train-step" || event.type === "train-start") &&
        event.experiment) {
      trainedTutorialExperiments.current.add(event.experiment);
      matched = trainedTutorialExperiments.current.size === 2;
      if (!matched) {
        setTutorialFeedback(
          `${event.feedback} Now train the other experiment as well.`,
        );
      }
    } else if (
      required === "change-architecture-b" &&
      event.type === "change-architecture" &&
      event.experiment === "B"
    ) {
      matched = true;
    } else if (
      required === "change-activation-b" &&
      event.type === "change-activation" &&
      event.experiment === "B"
    ) {
      matched = true;
    } else if (
      required === event.type &&
      (!event.experiment || event.experiment === "A")
    ) {
      matched = true;
    }

    if (matched) {
      setSegmentTaskComplete(true);
      setTutorialFeedback(event.feedback);
    }
  };

  const answerQuiz = (choice: number) => {
    const quiz = activeTutorialSegment.quiz;
    if (!quiz) return;
    setQuizChoice(choice);
    if (choice === quiz.answer) {
      setSegmentTaskComplete(true);
      setTutorialFeedback(quiz.explanation);
    } else {
      setSegmentTaskComplete(false);
      setTutorialFeedback(
        "Not quite. Re-read the plain explanation and try another answer.",
      );
    }
  };

  const prepareTutorialSegment = (index: number) => {
    const segment = courseSegments[index];
    setTutorialIndex(index);
    setSeenSegments((segments) =>
      segments.includes(index) ? segments : [...segments, index],
    );
    setShowWhy(false);
    setShowTechnical(false);
    setQuizChoice(null);
    setTutorialFeedback(null);
    setSegmentTaskComplete(!segment.action && !segment.quiz);
    trainedTutorialExperiments.current = new Set();
  };

  const beginTutorial = (index = 0) => {
    setDatasetKind("xor");
    setSeed(42);
    setSampleCount(120);
    setNoisePercent(0);
    setExperimentConfigs(initialConfigs);
    setPhysics(false);
    setTutorialRun((run) => run + 1);
    setSeenSegments([]);
    prepareTutorialSegment(index);
    setDismissedBonusFacts([]);
    setShowBonusStopPrompt(false);
    recentBonusDismissals.current = [];
    setTutorialActive(true);
    setGuideMenuOpen(false);
    setShowTutorialSegments(false);
  };

  const jumpToSegment = (index: number) => {
    if (!tutorialActive) {
      beginTutorial(index);
      return;
    }
    prepareTutorialSegment(index);
    setGuideMenuOpen(false);
    setShowTutorialSegments(false);
  };

  const endGuide = () => {
    setTutorialActive(false);
    setGuideMenuOpen(false);
    setShowTutorialSegments(false);
    showGuideMessage("Hope you had fun! Click again if needed!");
  };

  const skipSegment = () => {
    showGuideMessage("Got it!");
    if (tutorialIndex < courseSegments.length - 1) {
      prepareTutorialSegment(tutorialIndex + 1);
    } else {
      setTutorialActive(false);
    }
  };

  const nextSegment = () => {
    if (tutorialIndex < courseSegments.length - 1) {
      prepareTutorialSegment(tutorialIndex + 1);
    } else {
      endGuide();
    }
  };

  const applyChallenge = (
    kind: "minimal-xor" | "activation-duel" | "noisy-circles",
  ) => {
    setTutorialActive(false);
    setPhysics(false);
    if (kind === "minimal-xor") {
      setDatasetKind("xor");
      setNoisePercent(0);
      setExperimentConfigs([
        { architecture: "3", activation: "tanh", learningRate: 0.03 },
        { architecture: "6-4", activation: "tanh", learningRate: 0.03 },
      ]);
    } else if (kind === "activation-duel") {
      setDatasetKind("moons");
      setNoisePercent(0);
      setExperimentConfigs([
        { architecture: "6-4", activation: "tanh", learningRate: 0.03 },
        { architecture: "6-4", activation: "relu", learningRate: 0.03 },
      ]);
    } else {
      setDatasetKind("circles");
      setNoisePercent(15);
      setExperimentConfigs([
        { architecture: "3", activation: "tanh", learningRate: 0.03 },
        { architecture: "8-6-4", activation: "tanh", learningRate: 0.03 },
      ]);
    }
    setTutorialRun((run) => run + 1);
    document
      .querySelector(".lab-shell")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  useEffect(() => {
    const preferenceTimer = setTimeout(() => {
      const storedPreference = window.localStorage.getItem("nnvl-bonus-facts");
      if (storedPreference === "off") setBonusFactsEnabled(false);
    }, 0);
    return () => clearTimeout(preferenceTimer);
  }, []);

  useEffect(() => {
    if (!showBonusStopPrompt) return;
    const promptTimer = setTimeout(() => {
      setShowBonusStopPrompt(false);
      recentBonusDismissals.current = [];
    }, 7000);
    return () => clearTimeout(promptTimer);
  }, [showBonusStopPrompt]);

  useEffect(() => {
    if (!tutorialActive) return;
    const scrollTimer = setTimeout(() => {
      document
        .querySelector(`[data-guide-target="${activeTutorialSegment.target}"]`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 120);
    return () => clearTimeout(scrollTimer);
  }, [activeTutorialSegment.target, tutorialActive, tutorialIndex]);

  useEffect(
    () => () => {
      if (guideMessageTimer.current) clearTimeout(guideMessageTimer.current);
    },
    [],
  );

  return (
    <main className={tutorialActive ? "tutorial-running" : undefined}>
      <div className="guide-launcher-bar">
        <div className="guide-launcher">
          <button
            className="guide-launch-button"
            type="button"
            aria-expanded={guideMenuOpen}
            aria-controls="guide-menu"
            onClick={() => setGuideMenuOpen((open) => !open)}
          >
            <span className="guide-spark" aria-hidden="true">?</span>
            <span>Need a hand?</span>
            <span className="guide-chevron" aria-hidden="true">
              {guideMenuOpen ? "⌃" : "⌄"}
            </span>
          </button>

          {guideMenuOpen ? (
            <div className="guide-menu" id="guide-menu">
              <div className="guide-menu-actions">
                <button type="button" onClick={() => beginTutorial(0)}>
                  <span aria-hidden="true">▶</span>
                  <span><strong>{tutorialActive ? "Restart complete tutorial" : "Start complete tutorial"}</strong><small>Five chapters · no prior knowledge needed</small></span>
                </button>
                <button
                  type="button"
                  aria-expanded={showTutorialSegments}
                  onClick={() => setShowTutorialSegments((visible) => !visible)}
                >
                  <span aria-hidden="true">☷</span>
                  <span><strong>Show tutorial segments</strong><small>Jump to any concept</small></span>
                </button>
              </div>

              <button
                type="button"
                className="bonus-fact-toggle"
                role="switch"
                aria-checked={bonusFactsEnabled}
                onClick={() => setBonusFactPreference(!bonusFactsEnabled)}
              >
                <span>
                  <strong>Bonus facts</strong>
                  <small>Optional deeper details during the tutorial</small>
                </span>
                <i className={bonusFactsEnabled ? "active" : undefined} aria-hidden="true">
                  <b />
                </i>
                <em>{bonusFactsEnabled ? "On" : "Off"}</em>
              </button>

              {showTutorialSegments ? (
                <div className="tutorial-segment-list">
                  {([1, 2, 3, 4, 5] as const).map((chapter) => {
                    const chapterSegments = courseSegments.filter(
                      (segment) => segment.chapter === chapter,
                    );
                    return (
                      <section key={chapter}>
                        <h4>Chapter {chapter} · {chapterSegments[0].chapterTitle}</h4>
                        <ol>
                          {chapterSegments.map((segment) => {
                            const index = courseSegments.findIndex(
                              (candidate) => candidate.id === segment.id,
                            );
                            return (
                              <li key={segment.id}>
                                <button
                                  type="button"
                                  className={tutorialActive && tutorialIndex === index ? "current" : undefined}
                                  onClick={() => jumpToSegment(index)}
                                >
                                  <span>{String(index + 1).padStart(2, "0")}</span>
                                  <span>{segment.eyebrow}</span>
                                  {seenSegments.includes(index) ? <i aria-label="Visited">✓</i> : null}
                                </button>
                              </li>
                            );
                          })}
                        </ol>
                      </section>
                    );
                  })}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
      <section className="hero">
        <div className="hero-copy">
          <p className="kicker"><span /> A small machine-learning workbench</p>
          <h1>Neural Network <em>Visual Lab</em></h1>
          <p className="hero-intro">
            See what a small neural network learns, one training step at a time.
            Change the data, architecture, activation or learning rate, then compare
            the decision boundaries yourself.
          </p>
        </div>
        <aside className="try-card">
          <span className="try-number">01</span>
          <div>
            <p className="eyebrow">New to neural networks?</p>
            <h2>Learn from labelled dots to controlled experiments.</h2>
            <p>Five progressive chapters with plain explanations, optional technical depth and real actions.</p>
            <button type="button" className="hero-tutorial-button" onClick={() => beginTutorial(0)}>
              Start complete tutorial
            </button>
          </div>
        </aside>
      </section>

      <section className="lab-shell" aria-label="Neural network experiment controls">
        <div
          className="dataset-bar"
          data-guide-target="dataset"
          data-guide-active={activeGuideTarget === "dataset" ? "true" : undefined}
        >
          <div className="dataset-heading">
            <p className="eyebrow">Shared dataset</p>
            <h2>{datasetCopy[datasetKind].title}</h2>
            <p>{datasetCopy[datasetKind].note}</p>
            <small>{dataset.length} training examples · {testDataset.length} clean unseen test examples</small>
          </div>
          <div
            className="global-controls"
            data-guide-target="dataset-controls"
            data-guide-active={activeGuideTarget === "dataset-controls" ? "true" : undefined}
          >
            <label>
              Dataset
              <select
                value={datasetKind}
                onChange={(event) => setDatasetKind(event.target.value as DatasetKind)}
              >
                <option value="moons">Interleaving moons</option>
                <option value="xor">XOR quadrants</option>
                <option value="circles">Nested circles</option>
              </select>
            </label>
            <label>
              Seed
              <input
                type="number"
                min={1}
                max={9999}
                value={seed}
                onChange={(event) => {
                  const nextSeed = Math.max(1, Number(event.target.value) || 1);
                  setSeed(nextSeed);
                  handleTutorialAction({
                    type: "change-seed",
                    feedback: `Seed ${nextSeed} generated a new but reproducible set of examples and starting weights.`,
                  });
                }}
              />
            </label>
            <label>
              Samples
              <select
                value={sampleCount}
                onChange={(event) => setSampleCount(Number(event.target.value))}
              >
                <option value={120}>120</option>
                <option value={180}>180</option>
                <option value={260}>260</option>
              </select>
            </label>
            <label>
              Label noise
              <select
                value={noisePercent}
                onChange={(event) => {
                  const nextNoise = Number(event.target.value);
                  setNoisePercent(nextNoise);
                  handleTutorialAction({
                    type: "change-noise",
                    feedback: `${nextNoise}% label noise is now applied only to the training examples. The test set remains clean.`,
                  });
                }}
              >
                <option value={0}>0% · clean</option>
                <option value={5}>5% · light</option>
                <option value={15}>15% · visible</option>
                <option value={30}>30% · severe</option>
              </select>
            </label>
          </div>
        </div>

        {showModes ? <div
          className="mode-bar"
          data-guide-target="modes"
          data-guide-active={activeGuideTarget === "modes" ? "true" : undefined}
        >
          <div>
            <p className="eyebrow">Experimental modes</p>
            <p>Let the diagram move without changing the trained model.</p>
          </div>
          <div className="mode-buttons">
            <button
              className={physics ? "mode-button active" : "mode-button"}
              onClick={() => {
                const nextPhysics = !physics;
                setPhysics(nextPhysics);
                if (nextPhysics) {
                  handleTutorialAction({
                    type: "enable-mode",
                    feedback: "Node physics is enabled. Dragging nodes changes only the diagram layout, not the model's weights.",
                  });
                }
              }}
              aria-pressed={physics}
            >
              <span className="node-icon" aria-hidden="true">●—●</span>
              Node physics
            </button>
          </div>
        </div> : null}

        {showExperimentA ? <div
          className={showExperimentB ? "experiment-grid" : "experiment-grid single"}
          data-guide-target="comparison"
          data-guide-active={activeGuideTarget === "comparison" ? "true" : undefined}
        >
          {showExperimentA ? <Experiment
            key={`A-${tutorialRun}`}
            label="A"
            accent="violet"
            initialConfig={experimentConfigs[0]}
            dataset={dataset}
            testDataset={testDataset}
            seed={seed}
            physics={physics}
            guideTarget={activeGuideTarget}
            visibility={tutorialVisibility}
            onTutorialAction={handleTutorialAction}
          /> : null}
          {showExperimentB ? <Experiment
            key={`B-${tutorialRun}`}
            label="B"
            accent="blue"
            initialConfig={experimentConfigs[1]}
            dataset={dataset}
            testDataset={testDataset}
            seed={seed}
            physics={physics}
            guideTarget={null}
            visibility={tutorialVisibility}
            onTutorialAction={handleTutorialAction}
          /> : null}
        </div> : null}
      </section>

      <section className="explanation-section">
        <div>
          <p className="kicker"><span /> What is happening?</p>
          <h2>The colours are the model&apos;s confidence, not a prepared animation.</h2>
        </div>
        <div className="explanation-grid">
          <article>
            <span>01</span><h3>Forward pass</h3>
            <p>Each point travels through weighted connections and activation functions to produce a probability.</p>
          </article>
          <article>
            <span>02</span><h3>Loss</h3>
            <p>Binary cross-entropy measures how far the predictions are from the labelled examples.</p>
          </article>
          <article>
            <span>03</span><h3>Backpropagation</h3>
            <p>Gradients update every weight and bias. The boundary is redrawn after each training frame.</p>
          </article>
        </div>
      </section>

      <section className="challenge-section" aria-labelledby="challenge-title">
        <div className="challenge-heading">
          <p className="kicker"><span /> Challenge deck</p>
          <h2 id="challenge-title">Turn the lab into a question, not a collection of controls.</h2>
          <p>Each preset prepares a fair starting point. Predict the result before training, then use the maps and metrics to explain what happened.</p>
        </div>
        <div className="challenge-grid">
          <article>
            <span>01</span>
            <h3>Smallest useful XOR model</h3>
            <p>Can Compact match or beat the larger challenger without label noise?</p>
            <button type="button" onClick={() => applyChallenge("minimal-xor")}>Load XOR challenge</button>
          </article>
          <article>
            <span>02</span>
            <h3>Activation showdown</h3>
            <p>Hold architecture and learning rate constant while tanh faces ReLU.</p>
            <button type="button" onClick={() => applyChallenge("activation-duel")}>Load activation duel</button>
          </article>
          <article>
            <span>03</span>
            <h3>Overfit the noise</h3>
            <p>Compare Compact and Deep on noisy circles while the test set stays clean.</p>
            <button type="button" onClick={() => applyChallenge("noisy-circles")}>Load noisy challenge</button>
          </article>
        </div>
      </section>

      <NeuralWorkbench
        samples={dataset}
        guideActive={activeGuideTarget === "workbench"}
      />

      {tutorialActive ? (
        <div className="tutorial-panel-wrap">
          <div className="tutorial-side-actions" aria-label="Tutorial shortcuts">
            <button type="button" onClick={skipSegment}>Skip segment</button>
            <button type="button" onClick={endGuide}>End guide</button>
          </div>
          <aside className="tutorial-panel" aria-live="polite" aria-label="Neural network tutorial">
            <div className="tutorial-progress-row">
              <span>Chapter {activeTutorialSegment.chapter} · Segment {tutorialIndex + 1} of {courseSegments.length}</span>
              <span>{Math.round(((tutorialIndex + 1) / courseSegments.length) * 100)}%</span>
            </div>
            <div className="tutorial-progress" aria-hidden="true">
              <i style={{ width: `${((tutorialIndex + 1) / courseSegments.length) * 100}%` }} />
            </div>
            <p className="tutorial-chapter-name">{activeTutorialSegment.chapterTitle}</p>
            <p className="eyebrow">{activeTutorialSegment.eyebrow}</p>
            <h2>{activeTutorialSegment.title}</h2>
            <p>{activeTutorialSegment.body}</p>

            <div className="tutorial-depth-buttons">
              <button
                type="button"
                aria-expanded={showWhy}
                onClick={() => setShowWhy((visible) => !visible)}
              >
                {showWhy ? "Hide why it matters" : "Why does this matter?"}
              </button>
              <button
                type="button"
                aria-expanded={showTechnical}
                onClick={() => setShowTechnical((visible) => !visible)}
              >
                {showTechnical ? "Hide technical detail" : "Show technical detail"}
              </button>
            </div>

            {showWhy ? (
              <div className="tutorial-depth-card why-card">
                <strong>Why it matters</strong>
                <p>{activeTutorialSegment.why}</p>
              </div>
            ) : null}
            {showTechnical ? (
              <div className="tutorial-depth-card technical-card">
                <strong>Technical detail</strong>
                <p>{activeTutorialSegment.technical}</p>
              </div>
            ) : null}

            <div className="tutorial-try">
              <strong>Try this now</strong>
              <p>{activeTutorialSegment.tryThis}</p>
            </div>

            {activeTutorialSegment.quiz ? (
              <div className="tutorial-quiz">
                <strong>Quick check</strong>
                <p>{activeTutorialSegment.quiz.question}</p>
                <div>
                  {activeTutorialSegment.quiz.choices.map((choice, index) => (
                    <button
                      type="button"
                      key={choice}
                      className={quizChoice === index ? "selected" : undefined}
                      onClick={() => answerQuiz(index)}
                    >
                      <span>{String.fromCharCode(65 + index)}</span>{choice}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            {activeTutorialSegment.action ? (
              <div className={segmentTaskComplete ? "tutorial-task complete" : "tutorial-task"}>
                <span aria-hidden="true">{segmentTaskComplete ? "✓" : "→"}</span>
                <div>
                  <strong>{segmentTaskComplete ? "Action completed" : "Your action"}</strong>
                  <p>{activeTutorialSegment.action.prompt}</p>
                </div>
              </div>
            ) : null}

            {tutorialFeedback ? (
              <div className={segmentTaskComplete ? "tutorial-response success" : "tutorial-response"} role="status">
                {tutorialFeedback}
              </div>
            ) : null}

            <div className="tutorial-step-dots" aria-label="Tutorial progress">
              {activeChapterSegments.map((segment) => {
                const index = courseSegments.findIndex(
                  (candidate) => candidate.id === segment.id,
                );
                return (
                  <button
                    type="button"
                    key={segment.id}
                    className={index === tutorialIndex ? "current" : undefined}
                    aria-label={`Go to segment ${index + 1}: ${segment.eyebrow}`}
                    onClick={() => jumpToSegment(index)}
                  >
                    {index + 1}
                  </button>
                );
              })}
            </div>
            <div className="tutorial-navigation">
              <button
                type="button"
                className="secondary-button"
                disabled={tutorialIndex === 0}
                onClick={() => prepareTutorialSegment(Math.max(0, tutorialIndex - 1))}
              >
                Back
              </button>
              <button
                type="button"
                className="primary-button"
                disabled={!segmentTaskComplete}
                onClick={nextSegment}
              >
                {tutorialIndex === courseSegments.length - 1 ? "Finish guide" : "Continue"}
              </button>
            </div>
          </aside>
        </div>
      ) : null}

      {tutorialActive && bonusFactsEnabled ? (
        <div className="bonus-fact-stack" aria-live="polite">
          {!showBonusStopPrompt && !dismissedBonusFacts.includes(activeTutorialSegment.id) ? (
            <aside className="bonus-fact-card" key={activeTutorialSegment.id}>
              <header>
                <strong><span aria-hidden="true">✦</span> Bonus fact:</strong>
                <button
                  type="button"
                  aria-label="Dismiss bonus fact"
                  onClick={dismissBonusFact}
                >
                  ×
                </button>
              </header>
              <p>{activeTutorialSegment.bonusFact}</p>
            </aside>
          ) : null}

          {showBonusStopPrompt ? (
            <button
              type="button"
              className="bonus-stop-prompt"
              onClick={stopBonusFacts}
            >
              <span aria-hidden="true">×2</span>
              <strong>Click to stop bonus facts</strong>
            </button>
          ) : null}
        </div>
      ) : null}

      {guideMessage ? (
        <div className="guide-toast" role="status">{guideMessage}</div>
      ) : null}

      <footer>
        <p>Built by Writban Alim · Neural Network Visual Lab</p>
        <p>Everything trains in your browser · no prepared results</p>
      </footer>
    </main>
  );
}
