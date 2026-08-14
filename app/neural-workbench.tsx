"use client";

import {
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ActivationName, Sample } from "./neural-core";

type WorkNodeKind = "input" | "hidden" | "output";

type WorkNode = {
  id: string;
  label: string;
  kind: WorkNodeKind;
  x: number;
  y: number;
  bias: number;
  activation: ActivationName;
};

type WorkEdge = {
  id: string;
  from: string;
  to: string;
  weight: number;
  color: string;
};

type CanvasPoint = { x: number; y: number; time?: number };

type WorkbenchView = "fit" | "inputs" | "hidden" | "output";

const workbenchViews: Record<
  WorkbenchView,
  { label: string; x: number; y: number; width: number; height: number }
> = {
  fit: { label: "Fit all", x: 0, y: 0, width: 1100, height: 620 },
  inputs: { label: "Inputs", x: 0, y: 0, width: 620, height: 620 },
  hidden: { label: "Hidden", x: 245, y: 0, width: 620, height: 620 },
  output: { label: "Output", x: 480, y: 0, width: 620, height: 620 },
};

const edgeTones: Array<{
  color: string;
  name: string;
  description: string;
  waveform: OscillatorType;
  filterFrequency: number;
  pitchRatio: number;
}> = [
  { color: "#176b65", name: "Petrol", description: "clean", waveform: "sine", filterFrequency: 5200, pitchRatio: 1 },
  { color: "#b65a3a", name: "Rust", description: "bright", waveform: "sawtooth", filterFrequency: 4400, pitchRatio: 1.122 },
  { color: "#d4a13a", name: "Mustard", description: "soft", waveform: "triangle", filterFrequency: 3800, pitchRatio: 1.26 },
  { color: "#60727a", name: "Slate", description: "hollow", waveform: "square", filterFrequency: 3200, pitchRatio: 1.335 },
  { color: "#715b83", name: "Plum", description: "muted", waveform: "triangle", filterFrequency: 2600, pitchRatio: 1.498 },
];

type GraphPass = {
  valid: boolean;
  order: string[];
  values: Record<string, number>;
  weighted: Record<string, number>;
};

type Selection =
  | { kind: "node"; id: string }
  | { kind: "edge"; id: string }
  | null;

const starterNodes: WorkNode[] = [
  { id: "input-x", label: "x", kind: "input", x: 110, y: 205, bias: 0, activation: "tanh" },
  { id: "input-y", label: "y", kind: "input", x: 110, y: 415, bias: 0, activation: "tanh" },
  { id: "hidden-1", label: "h1", kind: "hidden", x: 470, y: 155, bias: 0, activation: "tanh" },
  { id: "hidden-2", label: "h2", kind: "hidden", x: 470, y: 310, bias: 0, activation: "tanh" },
  { id: "hidden-3", label: "h3", kind: "hidden", x: 470, y: 465, bias: 0, activation: "tanh" },
  { id: "output", label: "ŷ", kind: "output", x: 950, y: 310, bias: 0, activation: "sigmoid" },
];

const starterEdges: WorkEdge[] = [
  { id: "e-1", from: "input-x", to: "hidden-1", weight: 0.8, color: "#176b65" },
  { id: "e-2", from: "input-x", to: "hidden-2", weight: -0.65, color: "#b65a3a" },
  { id: "e-3", from: "input-x", to: "hidden-3", weight: 0.4, color: "#176b65" },
  { id: "e-4", from: "input-y", to: "hidden-1", weight: -0.5, color: "#b65a3a" },
  { id: "e-5", from: "input-y", to: "hidden-2", weight: 0.9, color: "#176b65" },
  { id: "e-6", from: "input-y", to: "hidden-3", weight: 0.55, color: "#176b65" },
  { id: "e-7", from: "hidden-1", to: "output", weight: 0.75, color: "#176b65" },
  { id: "e-8", from: "hidden-2", to: "output", weight: -0.8, color: "#b65a3a" },
  { id: "e-9", from: "hidden-3", to: "output", weight: 0.65, color: "#176b65" },
];

function segmentsCross(
  cursorStart: CanvasPoint,
  cursorEnd: CanvasPoint,
  edgeStart: CanvasPoint,
  edgeEnd: CanvasPoint,
) {
  const cross = (first: CanvasPoint, second: CanvasPoint, third: CanvasPoint) =>
    (second.x - first.x) * (third.y - first.y) -
    (second.y - first.y) * (third.x - first.x);
  const firstSide = cross(cursorStart, cursorEnd, edgeStart);
  const secondSide = cross(cursorStart, cursorEnd, edgeEnd);
  const thirdSide = cross(edgeStart, edgeEnd, cursorStart);
  const fourthSide = cross(edgeStart, edgeEnd, cursorEnd);
  const boundsOverlap =
    Math.max(Math.min(cursorStart.x, cursorEnd.x), Math.min(edgeStart.x, edgeEnd.x)) <=
      Math.min(Math.max(cursorStart.x, cursorEnd.x), Math.max(edgeStart.x, edgeEnd.x)) &&
    Math.max(Math.min(cursorStart.y, cursorEnd.y), Math.min(edgeStart.y, edgeEnd.y)) <=
      Math.min(Math.max(cursorStart.y, cursorEnd.y), Math.max(edgeStart.y, edgeEnd.y));
  return boundsOverlap && firstSide * secondSide <= 0 && thirdSide * fourthSide <= 0;
}

function activate(value: number, activation: ActivationName) {
  if (activation === "relu") return Math.max(0, value);
  if (activation === "sigmoid") return 1 / (1 + Math.exp(-value));
  return Math.tanh(value);
}

function activationDerivative(
  weighted: number,
  value: number,
  activation: ActivationName,
) {
  if (activation === "relu") return weighted > 0 ? 1 : 0;
  if (activation === "sigmoid") return value * (1 - value);
  return 1 - value * value;
}

function orderedInputs(nodes: WorkNode[]) {
  return nodes
    .filter((node) => node.kind === "input")
    .slice()
    .sort((left, right) => left.y - right.y);
}

function topology(nodes: WorkNode[], edges: WorkEdge[]) {
  const ids = new Set(nodes.map((node) => node.id));
  const indegree = new Map(nodes.map((node) => [node.id, 0]));
  const outgoing = new Map(nodes.map((node) => [node.id, [] as string[]]));
  edges.forEach((edge) => {
    if (!ids.has(edge.from) || !ids.has(edge.to)) return;
    indegree.set(edge.to, (indegree.get(edge.to) ?? 0) + 1);
    outgoing.get(edge.from)?.push(edge.to);
  });
  const queue = nodes
    .filter((node) => (indegree.get(node.id) ?? 0) === 0)
    .map((node) => node.id);
  const order: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift();
    if (!id) break;
    order.push(id);
    outgoing.get(id)?.forEach((target) => {
      const next = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, next);
      if (next === 0) queue.push(target);
    });
  }
  return { valid: order.length === nodes.length, order };
}

function runGraph(
  nodes: WorkNode[],
  edges: WorkEdge[],
  inputValues: Record<string, number>,
): GraphPass {
  const sorted = topology(nodes, edges);
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const values: Record<string, number> = {};
  const weighted: Record<string, number> = {};
  if (!sorted.valid) return { ...sorted, values, weighted };

  sorted.order.forEach((id) => {
    const node = nodeMap.get(id);
    if (!node) return;
    if (node.kind === "input") {
      values[id] = inputValues[id] ?? 0;
      weighted[id] = values[id];
      return;
    }
    const total = edges
      .filter((edge) => edge.to === id)
      .reduce(
        (sum, edge) => sum + edge.weight * (values[edge.from] ?? 0),
        node.bias,
      );
    weighted[id] = total;
    values[id] = activate(total, node.kind === "output" ? "sigmoid" : node.activation);
  });
  return { ...sorted, values, weighted };
}

function sampleInputs(nodes: WorkNode[], sample: Sample) {
  const inputs = orderedInputs(nodes);
  return Object.fromEntries(
    inputs.map((node, index) => [
      node.id,
      index === 0 ? sample.x : index === 1 ? sample.y : 0,
    ]),
  );
}

function graphMetrics(nodes: WorkNode[], edges: WorkEdge[], samples: Sample[]) {
  const output = nodes.find((node) => node.kind === "output");
  if (!output || samples.length === 0 || !topology(nodes, edges).valid) {
    return { loss: Number.NaN, accuracy: Number.NaN };
  }
  let loss = 0;
  let correct = 0;
  samples.forEach((sample) => {
    const prediction = runGraph(nodes, edges, sampleInputs(nodes, sample)).values[output.id] ?? 0.5;
    const clipped = Math.min(1 - 1e-7, Math.max(1e-7, prediction));
    loss += -sample.label * Math.log(clipped) - (1 - sample.label) * Math.log(1 - clipped);
    if ((prediction >= 0.5 ? 1 : 0) === sample.label) correct += 1;
  });
  return { loss: loss / samples.length, accuracy: correct / samples.length };
}

function createsCycle(edges: WorkEdge[], from: string, to: string) {
  const outgoing = new Map<string, string[]>();
  [...edges, { id: "candidate", from, to, weight: 0, color: "#176b65" }].forEach((edge) => {
    outgoing.set(edge.from, [...(outgoing.get(edge.from) ?? []), edge.to]);
  });
  const stack = [to];
  const seen = new Set<string>();
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current || seen.has(current)) continue;
    if (current === from) return true;
    seen.add(current);
    stack.push(...(outgoing.get(current) ?? []));
  }
  return false;
}

function trainGraph(
  originalNodes: WorkNode[],
  originalEdges: WorkEdge[],
  samples: Sample[],
  steps: number,
) {
  const nodes = originalNodes.map((node) => ({ ...node }));
  const edges = originalEdges.map((edge) => ({ ...edge }));
  const output = nodes.find((node) => node.kind === "output");
  if (!output || !topology(nodes, edges).valid || samples.length === 0) {
    return { nodes, edges };
  }

  for (let step = 0; step < steps; step += 1) {
    const edgeGradients = new Map(edges.map((edge) => [edge.id, 0]));
    const biasGradients = new Map(nodes.map((node) => [node.id, 0]));
    const order = topology(nodes, edges).order;

    samples.forEach((sample) => {
      const pass = runGraph(nodes, edges, sampleInputs(nodes, sample));
      const deltas: Record<string, number> = {
        [output.id]: (pass.values[output.id] ?? 0.5) - sample.label,
      };
      [...order].reverse().forEach((id) => {
        const node = nodes.find((candidate) => candidate.id === id);
        if (!node || node.kind !== "hidden") return;
        const downstream = edges
          .filter((edge) => edge.from === id)
          .reduce(
            (sum, edge) => sum + edge.weight * (deltas[edge.to] ?? 0),
            0,
          );
        deltas[id] =
          downstream *
          activationDerivative(
            pass.weighted[id] ?? 0,
            pass.values[id] ?? 0,
            node.activation,
          );
      });
      edges.forEach((edge) => {
        edgeGradients.set(
          edge.id,
          (edgeGradients.get(edge.id) ?? 0) +
            (deltas[edge.to] ?? 0) * (pass.values[edge.from] ?? 0),
        );
      });
      nodes.forEach((node) => {
        if (node.kind !== "input") {
          biasGradients.set(
            node.id,
            (biasGradients.get(node.id) ?? 0) + (deltas[node.id] ?? 0),
          );
        }
      });
    });

    const scale = 0.045 / samples.length;
    edges.forEach((edge) => {
      edge.weight -= scale * (edgeGradients.get(edge.id) ?? 0);
    });
    nodes.forEach((node) => {
      if (node.kind !== "input") {
        node.bias -= scale * (biasGradients.get(node.id) ?? 0);
      }
    });
  }
  return { nodes, edges };
}

export default function NeuralWorkbench({
  samples,
  guideActive = false,
}: {
  samples: Sample[];
  guideActive?: boolean;
}) {
  const sectionRef = useRef<HTMLElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const audioRef = useRef<AudioContext | null>(null);
  const audioOutputRef = useRef<AudioNode | null>(null);
  const idRef = useRef(20);
  const strumRef = useRef<{
    active: boolean;
    pointerId: number | null;
    lastPoint: CanvasPoint | null;
    struckIds: string[];
  }>({ active: false, pointerId: null, lastPoint: null, struckIds: [] });
  const lastStruckRef = useRef<Record<string, number>>({});
  const [nodes, setNodes] = useState<WorkNode[]>(starterNodes);
  const [edges, setEdges] = useState<WorkEdge[]>(starterEdges);
  const [selection, setSelection] = useState<Selection>(null);
  const [connectMode, setConnectMode] = useState(false);
  const [connectFrom, setConnectFrom] = useState<string | null>(null);
  const [dragging, setDragging] = useState<{ id: string; dx: number; dy: number } | null>(null);
  const [inputValues, setInputValues] = useState<Record<string, number>>({
    "input-x": 0.35,
    "input-y": -0.25,
  });
  const [values, setValues] = useState<Record<string, number>>({});
  const [pulse, setPulse] = useState(0);
  const [status, setStatus] = useState("Ready for a manual pulse.");
  const [instrumentUnlocked, setInstrumentUnlocked] = useState(false);
  const [colourEditing, setColourEditing] = useState(false);
  const [showBoredPrompt, setShowBoredPrompt] = useState(false);
  const [showStringHelp, setShowStringHelp] = useState(false);
  const [struckEdges, setStruckEdges] = useState<string[]>([]);
  const [workbenchView, setWorkbenchView] = useState<WorkbenchView>("fit");
  const activeWorkbenchView = workbenchViews[workbenchView];
  const metrics = useMemo(
    () => graphMetrics(nodes, edges, samples),
    [edges, nodes, samples],
  );
  const inputs = useMemo(() => orderedInputs(nodes), [nodes]);
  const selectedNode =
    selection?.kind === "node"
      ? nodes.find((node) => node.id === selection.id) ?? null
      : null;
  const selectedEdge =
    selection?.kind === "edge"
      ? edges.find((edge) => edge.id === selection.id) ?? null
      : null;

  const evaluateCurrent = useCallback(
    (nextInputs = inputValues) => {
      const pass = runGraph(nodes, edges, nextInputs);
      setValues(pass.values);
      setPulse((value) => value + 1);
      if (!pass.valid) setStatus("This graph contains a loop, so it cannot run as a feed-forward network.");
      return pass;
    },
    [edges, inputValues, nodes],
  );

  useEffect(() => {
    const timer = setTimeout(() => evaluateCurrent(), 0);
    return () => clearTimeout(timer);
  }, [evaluateCurrent]);

  const pointFromPointer = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix) return { x: 0, y: 0 };
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const transformed = point.matrixTransform(matrix.inverse());
    return { x: transformed.x, y: transformed.y };
  };

  const addNode = (kind: "input" | "hidden", point?: { x: number; y: number }) => {
    idRef.current += 1;
    const id = `${kind}-${idRef.current}`;
    const sameKindCount = nodes.filter((node) => node.kind === kind).length;
    const node: WorkNode = {
      id,
      label: kind === "input" ? `i${sameKindCount + 1}` : `h${sameKindCount + 1}`,
      kind,
      x: point?.x ?? (kind === "input" ? 120 : 560),
      y: point?.y ?? 120 + ((sameKindCount * 95) % 400),
      bias: 0,
      activation: "tanh",
    };
    setNodes((items) => [...items, node]);
    setInputValues((current) =>
      kind === "input" ? { ...current, [id]: 0 } : current,
    );
    setSelection({ kind: "node", id });
    setStatus(`${node.label} added. Connect it when you are ready.`);
  };

  const connectNodes = (from: string, to: string) => {
    const source = nodes.find((node) => node.id === from);
    const target = nodes.find((node) => node.id === to);
    if (!source || !target || source.kind === "output" || target.kind === "input") {
      setStatus("Connections must flow from an input or hidden neuron towards a hidden or output neuron.");
      return;
    }
    if (from === to || edges.some((edge) => edge.from === from && edge.to === to)) {
      setStatus("That connection already exists.");
      return;
    }
    if (createsCycle(edges, from, to)) {
      setStatus("That wire would create a loop. This workbench currently runs feed-forward networks only.");
      return;
    }
    idRef.current += 1;
    const edge = {
      id: `edge-${idRef.current}`,
      from,
      to,
      weight: 0.5,
      color: "#176b65",
    };
    setEdges((items) => [...items, edge]);
    setSelection({ kind: "edge", id: edge.id });
    setStatus(`Connected ${source.label} to ${target.label}.`);
  };

  const handleNodePointerDown = (
    event: ReactPointerEvent<SVGGElement>,
    node: WorkNode,
  ) => {
    event.stopPropagation();
    if (connectMode) {
      if (!connectFrom) {
        setConnectFrom(node.id);
        setStatus(`Starting at ${node.label}. Choose the destination neuron.`);
      } else {
        connectNodes(connectFrom, node.id);
        setConnectFrom(null);
        setConnectMode(false);
      }
      return;
    }
    const point = pointFromPointer(event.clientX, event.clientY);
    setDragging({ id: node.id, dx: node.x - point.x, dy: node.y - point.y });
    setSelection({ kind: "node", id: node.id });
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const pluckEdge = useCallback((edge: WorkEdge, length: number, speed: number) => {
    const context = audioRef.current;
    if (!context) return;
    const tone = edgeTones.find((candidate) => candidate.color === edge.color) ?? edgeTones[0];
    const speedStrength = Math.max(0.35, Math.min(1, speed / 2.1));
    const lengthFrequency = 126000 / Math.max(120, length);
    const frequency = Math.max(85, Math.min(1100, lengthFrequency * tone.pitchRatio));
    const duration = 0.42 + Math.min(0.4, length / 1500);
    const now = context.currentTime;
    const oscillator = context.createOscillator();
    const fundamental = context.createOscillator();
    const fundamentalGain = context.createGain();
    const filter = context.createBiquadFilter();
    const gain = context.createGain();
    oscillator.type = tone.waveform;
    oscillator.frequency.setValueAtTime(frequency, now);
    fundamental.type = "sine";
    fundamental.frequency.setValueAtTime(frequency, now);
    fundamental.detune.setValueAtTime(-5, now);
    fundamentalGain.gain.value = 0.38;
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(
      tone.filterFrequency * (0.82 + speedStrength * 0.36),
      now,
    );
    filter.Q.value = 0.8;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(
      0.07 + speedStrength * 0.18,
      now + 0.009,
    );
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
    oscillator.connect(filter);
    fundamental.connect(fundamentalGain);
    fundamentalGain.connect(filter);
    filter.connect(gain);
    gain.connect(audioOutputRef.current ?? context.destination);
    oscillator.start(now);
    fundamental.start(now);
    oscillator.stop(now + duration + 0.03);
    fundamental.stop(now + duration + 0.03);
  }, []);

  const beginStrum = (event: ReactPointerEvent<SVGElement>) => {
    if (!instrumentUnlocked || event.button !== 0) return;
    if (audioRef.current?.state === "suspended") void audioRef.current.resume();
    const point = pointFromPointer(event.clientX, event.clientY);
    strumRef.current = {
      active: true,
      pointerId: event.pointerId,
      lastPoint: { ...point, time: event.timeStamp },
      struckIds: [],
    };
    svgRef.current?.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const point = pointFromPointer(event.clientX, event.clientY);
    if (dragging) {
      setNodes((items) =>
        items.map((node) =>
          node.id === dragging.id
            ? {
                ...node,
                x: Math.max(45, Math.min(1055, point.x + dragging.dx)),
                y: Math.max(55, Math.min(565, point.y + dragging.dy)),
              }
            : node,
        ),
      );
      return;
    }

    const strum = strumRef.current;
    if (
      !instrumentUnlocked ||
      !strum.active ||
      !strum.lastPoint ||
      strum.pointerId !== event.pointerId
    ) return;
    const now = event.timeStamp;
    const currentPoint = { ...point, time: now };
    const travelled = Math.hypot(
      currentPoint.x - strum.lastPoint.x,
      currentPoint.y - strum.lastPoint.y,
    );
    const elapsed = Math.max(1, now - (strum.lastPoint.time ?? now));
    const speed = travelled / elapsed;
    const crossed: string[] = [];

    edges.forEach((edge) => {
      const source = nodes.find((node) => node.id === edge.from);
      const target = nodes.find((node) => node.id === edge.to);
      if (!source || !target) return;
      if (
        !segmentsCross(strum.lastPoint as CanvasPoint, currentPoint, source, target) ||
        now - (lastStruckRef.current[edge.id] ?? 0) < 85
      ) {
        return;
      }
      lastStruckRef.current[edge.id] = now;
      const length = Math.hypot(target.x - source.x, target.y - source.y);
      pluckEdge(edge, length, speed);
      crossed.push(edge.id);
    });

    if (crossed.length > 0) {
      strum.struckIds = [...new Set([...strum.struckIds, ...crossed])];
      setStatus(
        strum.struckIds.length > 1
          ? `${strum.struckIds.length} connection strings struck in this stroke.`
          : "Connection string struck.",
      );
      setStruckEdges((current) => [...new Set([...current, ...crossed])]);
      crossed.forEach((id) => {
        window.setTimeout(
          () => setStruckEdges((current) => current.filter((edgeId) => edgeId !== id)),
          270,
        );
      });
    }
    strumRef.current.lastPoint = currentPoint;
  };

  const endPointerAction = () => {
    setDragging(null);
    strumRef.current = { active: false, pointerId: null, lastPoint: null, struckIds: [] };
  };

  const deleteSelection = () => {
    if (!selection) return;
    if (selection.kind === "edge") {
      setEdges((items) => items.filter((edge) => edge.id !== selection.id));
      setStatus("Connection removed.");
    } else {
      const node = nodes.find((candidate) => candidate.id === selection.id);
      if (!node || node.kind === "output") {
        setStatus("Keep the output neuron so the network has something to predict.");
        return;
      }
      setNodes((items) => items.filter((candidate) => candidate.id !== selection.id));
      setEdges((items) =>
        items.filter((edge) => edge.from !== selection.id && edge.to !== selection.id),
      );
      setStatus(`${node.label} and its connections were removed.`);
    }
    setSelection(null);
  };

  const resetGraph = () => {
    setNodes(starterNodes.map((node) => ({ ...node })));
    setEdges(starterEdges.map((edge) => ({ ...edge })));
    setInputValues({ "input-x": 0.35, "input-y": -0.25 });
    setSelection(null);
    setConnectFrom(null);
    setConnectMode(false);
    setColourEditing(false);
    setStatus("Starter network restored.");
  };

  const train = () => {
    const trained = trainGraph(nodes, edges, samples, 40);
    setNodes(trained.nodes);
    setEdges(trained.edges);
    setStatus("Trained for 40 steps on the shared dataset. Test accuracy has been recalculated.");
  };

  const testDataset = () => {
    evaluateCurrent();
    setStatus(
      Number.isFinite(metrics.loss)
        ? `Shared dataset test: ${Math.round(metrics.accuracy * 100)}% correct, loss ${metrics.loss.toFixed(3)}.`
        : "The graph needs a valid path to the output before it can be tested.",
    );
  };

  const unlockStrings = async () => {
    if (!audioRef.current) {
      const AudioContextClass =
        window.AudioContext ??
        (window as typeof window & { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (AudioContextClass) {
        const context = new AudioContextClass();
        const compressor = context.createDynamicsCompressor();
        const masterGain = context.createGain();
        compressor.threshold.value = -18;
        compressor.knee.value = 16;
        compressor.ratio.value = 8;
        compressor.attack.value = 0.003;
        compressor.release.value = 0.18;
        masterGain.gain.value = 1.15;
        compressor.connect(masterGain);
        masterGain.connect(context.destination);
        audioRef.current = context;
        audioOutputRef.current = compressor;
      }
    }
    if (audioRef.current?.state === "suspended") await audioRef.current.resume();
    setInstrumentUnlocked(true);
    setShowBoredPrompt(false);
    window.sessionStorage.setItem("nnvl-bored-prompt", "used");
    setStatus("String mode unlocked. Hold the pointer down and drag across the connections.");
  };

  useEffect(() => {
    if (!instrumentUnlocked) return;
    const helpTimer = window.setTimeout(() => setShowStringHelp(true), 15000);
    return () => window.clearTimeout(helpTimer);
  }, [instrumentUnlocked]);

  useEffect(() => {
    const section = sectionRef.current;
    if (!section || instrumentUnlocked) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (timer) clearTimeout(timer);
        if (
          entry.isIntersecting &&
          entry.intersectionRatio >= 0.45 &&
          window.sessionStorage.getItem("nnvl-bored-prompt") !== "used"
        ) {
          timer = setTimeout(() => {
            setShowBoredPrompt(true);
            window.sessionStorage.setItem("nnvl-bored-prompt", "used");
          }, 8000);
        }
      },
      { threshold: [0.45] },
    );
    observer.observe(section);
    return () => {
      if (timer) clearTimeout(timer);
      observer.disconnect();
    };
  }, [instrumentUnlocked]);

  useEffect(
    () => () => {
      if (audioRef.current) void audioRef.current.close();
      audioOutputRef.current = null;
    },
    [],
  );

  return (
    <section className="workbench-section" id="neural-workbench" ref={sectionRef}>
      <header
        className="workbench-heading"
        data-guide-target="workbench"
        data-guide-active={guideActive ? "true" : undefined}
      >
        <div>
          <p className="eyebrow">Open bench · build it yourself</p>
          <h2>Neural Workbench</h2>
        </div>
        <p>
          Add neurons, draw individual connections, change weights and biases,
          then train or test the graph against the dataset above. Use the
          toolbar on a phone, or double-click empty space with a mouse, to add a
          hidden neuron.
        </p>
      </header>

      <div className="workbench-readout" aria-live="polite">
        <span>{nodes.length} nodes</span>
        <span>{edges.length} connections</span>
        <span>
          Accuracy {Number.isFinite(metrics.accuracy) ? `${Math.round(metrics.accuracy * 100)}%` : "—"}
        </span>
        <span>Loss {Number.isFinite(metrics.loss) ? metrics.loss.toFixed(3) : "—"}</span>
        <strong>{status}</strong>
      </div>

      <div className="workbench-shell">
        <div className="workbench-toolbar" aria-label="Neural Workbench tools">
          <button type="button" onClick={() => addNode("input")}>+ Input</button>
          <button type="button" onClick={() => addNode("hidden")}>+ Neuron</button>
          <button
            type="button"
            className={connectMode ? "active" : undefined}
            onClick={() => {
              setConnectMode((value) => !value);
              setConnectFrom(null);
              setStatus("Connection tool ready. Choose a source and destination neuron.");
            }}
          >
            {connectMode ? "Cancel wire" : "+ Connection"}
          </button>
          <button type="button" onClick={deleteSelection} disabled={!selection}>Delete selected</button>
          <span />
          <button type="button" className="workbench-train" onClick={train}>Train 40 steps</button>
          <button type="button" onClick={testDataset}>Test dataset</button>
          <button type="button" onClick={() => evaluateCurrent()}>Run pulse</button>
          <button type="button" onClick={resetGraph}>Reset bench</button>
          {instrumentUnlocked ? (
            <button
              type="button"
              className={colourEditing ? "active" : undefined}
              onClick={() => {
                setColourEditing((value) => !value);
                setStatus(
                  colourEditing
                    ? "String colour editor closed."
                    : "Colour editor ready. Select a connection, then choose its tone.",
                );
              }}
            >
              {colourEditing ? "Close colours" : "Edit edge colours"}
            </button>
          ) : null}
        </div>

        <div className="workbench-body">
          <div className="workbench-canvas-wrap">
            <div className="workbench-mobile-view" aria-label="Choose a workbench canvas view">
              <span>Canvas view</span>
              {(Object.keys(workbenchViews) as WorkbenchView[]).map((view) => (
                <button
                  type="button"
                  key={view}
                  className={workbenchView === view ? "active" : undefined}
                  aria-pressed={workbenchView === view}
                  onClick={() => setWorkbenchView(view)}
                >
                  {workbenchViews[view].label}
                </button>
              ))}
            </div>
            <svg
              ref={svgRef}
              className={`workbench-canvas ${instrumentUnlocked ? "string-mode" : ""}`}
              viewBox={`${activeWorkbenchView.x} ${activeWorkbenchView.y} ${activeWorkbenchView.width} ${activeWorkbenchView.height}`}
              role="img"
              aria-label="Editable feed-forward neural network. Drag neurons or use the toolbar and inspector to edit the graph."
              onPointerMove={handlePointerMove}
              onPointerUp={endPointerAction}
              onPointerCancel={endPointerAction}
              onPointerDown={(event) => {
                setSelection(null);
                beginStrum(event);
              }}
              onDoubleClick={(event) => {
                if (instrumentUnlocked) return;
                const point = pointFromPointer(event.clientX, event.clientY);
                addNode("hidden", point);
              }}
            >
              <defs>
                <pattern id="bench-grid" width="28" height="28" patternUnits="userSpaceOnUse">
                  <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(24,35,33,0.075)" strokeWidth="1" />
                </pattern>
                <marker id="edge-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8 Z" fill="context-stroke" />
                </marker>
              </defs>
              <rect width="1100" height="620" fill="url(#bench-grid)" />
              <text x="70" y="48" className="bench-column-label">INPUTS</text>
              <text x="455" y="48" className="bench-column-label">HIDDEN SPACE</text>
              <text x="925" y="48" className="bench-column-label">OUTPUT</text>

              {edges.map((edge) => {
                const source = nodes.find((node) => node.id === edge.from);
                const target = nodes.find((node) => node.id === edge.to);
                if (!source || !target) return null;
                const selected = selection?.kind === "edge" && selection.id === edge.id;
                const struck = struckEdges.includes(edge.id);
                return (
                  <g key={edge.id}>
                    <line
                      x1={source.x}
                      y1={source.y}
                      x2={target.x}
                      y2={target.y}
                      className="work-edge-hit"
                      stroke="transparent"
                      strokeWidth="28"
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        setSelection({ kind: "edge", id: edge.id });
                        beginStrum(event);
                      }}
                    />
                    <line
                      x1={source.x}
                      y1={source.y}
                      x2={target.x}
                      y2={target.y}
                      className={`work-edge ${selected ? "selected" : ""} ${struck ? "struck" : ""}`}
                      stroke={edge.color}
                      strokeWidth={1.5 + Math.min(4, Math.abs(edge.weight) * 1.25)}
                      markerEnd="url(#edge-arrow)"
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        setSelection({ kind: "edge", id: edge.id });
                        beginStrum(event);
                      }}
                    />
                    <text
                      x={(source.x + target.x) / 2}
                      y={(source.y + target.y) / 2 - 7}
                      className="edge-weight-label"
                    >
                      {edge.weight.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {nodes.map((node) => {
                const value = values[node.id] ?? 0;
                const strength = Math.min(1, Math.abs(value));
                const selected = selection?.kind === "node" && selection.id === node.id;
                const connecting = connectFrom === node.id;
                const fill =
                  node.kind === "input"
                    ? "#d4a13a"
                    : node.kind === "output"
                      ? "#b65a3a"
                      : `hsl(176 45% ${76 - strength * 34}%)`;
                return (
                  <g
                    key={node.id}
                    className={`work-node ${selected ? "selected" : ""} ${connecting ? "connecting" : ""}`}
                    onPointerDown={(event) => handleNodePointerDown(event, node)}
                  >
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r="44"
                      className="node-hit-target"
                    />
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={24 + strength * 6}
                      fill={fill}
                      className={pulse % 2 === 0 ? "node-pulse-even" : "node-pulse-odd"}
                    />
                    <text x={node.x} y={node.y + 5} textAnchor="middle" className="node-label">
                      {node.label}
                    </text>
                    <text x={node.x} y={node.y + 48} textAnchor="middle" className="node-value">
                      {value.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </svg>

            {showBoredPrompt ? (
              <div className="bored-note">
                <button type="button" className="bored-dismiss" aria-label="Dismiss suggestion" onClick={() => setShowBoredPrompt(false)}>×</button>
                <button type="button" className="bored-unlock" onClick={() => void unlockStrings()}>
                  Feeling bored? Click here.
                </button>
              </div>
            ) : null}

            {showStringHelp ? (
              <aside className="string-help-bubble" role="status">
                <button type="button" aria-label="Dismiss string instructions" onClick={() => setShowStringHelp(false)}>×</button>
                <strong>The connections are strings now.</strong>
                <p>
                  Hold the pointer down and drag across them. Crossing several in one stroke plays them all, and faster strokes sound stronger. Connection length works like string length: a shorter edge vibrates faster and plays a higher note, while a longer edge vibrates more slowly and plays a lower note. Drag either neuron at the end of an edge to change its length and retune it. Use “Edit edge colours” to change its note and timbre further.
                </p>
              </aside>
            ) : null}
          </div>

          <aside className="workbench-inspector">
            <section>
              <p className="eyebrow">Manual inputs</p>
              {inputs.length === 0 ? <p>Add an input node to run a pulse.</p> : null}
              {inputs.map((node) => (
                <label key={node.id}>
                  {node.label} <output>{(inputValues[node.id] ?? 0).toFixed(2)}</output>
                  <input
                    type="range"
                    min={-1}
                    max={1}
                    step={0.05}
                    value={inputValues[node.id] ?? 0}
                    onChange={(event) => {
                      const nextInputs = { ...inputValues, [node.id]: Number(event.target.value) };
                      setInputValues(nextInputs);
                      evaluateCurrent(nextInputs);
                    }}
                  />
                </label>
              ))}
            </section>

            <section className="selection-inspector">
              <p className="eyebrow">Selected component</p>
              {!selectedNode && !selectedEdge ? (
                <p>Select a neuron or connection to edit its properties.</p>
              ) : null}
              {selectedNode ? (
                <>
                  <strong>{selectedNode.label} · {selectedNode.kind}</strong>
                  {selectedNode.kind !== "input" ? (
                    <label>
                      Bias <output>{selectedNode.bias.toFixed(2)}</output>
                      <input
                        type="range"
                        min={-2}
                        max={2}
                        step={0.05}
                        value={selectedNode.bias}
                        onChange={(event) =>
                          setNodes((items) =>
                            items.map((node) =>
                              node.id === selectedNode.id
                                ? { ...node, bias: Number(event.target.value) }
                                : node,
                            ),
                          )
                        }
                      />
                    </label>
                  ) : null}
                  {selectedNode.kind === "hidden" ? (
                    <label>
                      Activation
                      <select
                        value={selectedNode.activation}
                        onChange={(event) =>
                          setNodes((items) =>
                            items.map((node) =>
                              node.id === selectedNode.id
                                ? { ...node, activation: event.target.value as ActivationName }
                                : node,
                            ),
                          )
                        }
                      >
                        <option value="tanh">Tanh</option>
                        <option value="relu">ReLU</option>
                        <option value="sigmoid">Sigmoid</option>
                      </select>
                    </label>
                  ) : null}
                </>
              ) : null}
              {selectedEdge ? (
                <>
                  <strong>Connection weight</strong>
                  <label>
                    Weight <output>{selectedEdge.weight.toFixed(2)}</output>
                    <input
                      type="range"
                      min={-3}
                      max={3}
                      step={0.05}
                      value={selectedEdge.weight}
                      onChange={(event) =>
                        setEdges((items) =>
                          items.map((edge) =>
                            edge.id === selectedEdge.id
                              ? { ...edge, weight: Number(event.target.value) }
                              : edge,
                          ),
                        )
                      }
                    />
                  </label>
                  {instrumentUnlocked && colourEditing ? (
                    <fieldset className="edge-tone-picker">
                      <legend>String colour and tone</legend>
                      <div>
                        {edgeTones.map((tone) => (
                          <button
                            type="button"
                            key={tone.color}
                            className={selectedEdge.color === tone.color ? "selected" : undefined}
                            aria-label={`${tone.name}, ${tone.description} tone`}
                            title={`${tone.name}: ${tone.description} tone`}
                            style={{ backgroundColor: tone.color }}
                            onClick={() => {
                              setEdges((items) =>
                                items.map((edge) =>
                                  edge.id === selectedEdge.id
                                    ? { ...edge, color: tone.color }
                                    : edge,
                                ),
                              );
                              const source = nodes.find((node) => node.id === selectedEdge.from);
                              const target = nodes.find((node) => node.id === selectedEdge.to);
                              if (source && target) {
                                const preview = () => {
                                  pluckEdge(
                                    { ...selectedEdge, color: tone.color },
                                    Math.hypot(target.x - source.x, target.y - source.y),
                                    1.8,
                                  );
                                  setStatus(`${tone.name} tone selected and previewed.`);
                                };
                                if (audioRef.current?.state === "suspended") {
                                  void audioRef.current.resume().then(preview);
                                } else {
                                  preview();
                                }
                              }
                            }}
                          />
                        ))}
                      </div>
                      <small>Colour changes the note, waveform and timbre. Choosing a colour plays a preview. Weight remains part of the neural calculation.</small>
                    </fieldset>
                  ) : instrumentUnlocked ? (
                    <small>Use “Edit edge colours” above to change this string’s tone.</small>
                  ) : (
                    <small>Line width reflects the connection’s absolute weight.</small>
                  )}
                </>
              ) : null}
            </section>

            {instrumentUnlocked ? (
              <section className="string-mode-panel">
                <p className="eyebrow">String mode unlocked</p>
                <strong>Press, drag and cross a connection</strong>
                <p>Shorter edges play higher notes; longer edges play lower ones. Move either connected neuron to retune an edge, and use “Edit edge colours” above to change its note and timbre further.</p>
              </section>
            ) : null}
          </aside>
        </div>
      </div>
      <p className="workbench-note">
        Lab note: the editor prevents circular connections so its graphs remain trainable feed-forward networks. The output neuron uses a sigmoid; hidden neurons can be changed individually.
      </p>
    </section>
  );
}
