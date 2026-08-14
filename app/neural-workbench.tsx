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
};

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
  { id: "e-1", from: "input-x", to: "hidden-1", weight: 0.8 },
  { id: "e-2", from: "input-x", to: "hidden-2", weight: -0.65 },
  { id: "e-3", from: "input-x", to: "hidden-3", weight: 0.4 },
  { id: "e-4", from: "input-y", to: "hidden-1", weight: -0.5 },
  { id: "e-5", from: "input-y", to: "hidden-2", weight: 0.9 },
  { id: "e-6", from: "input-y", to: "hidden-3", weight: 0.55 },
  { id: "e-7", from: "hidden-1", to: "output", weight: 0.75 },
  { id: "e-8", from: "hidden-2", to: "output", weight: -0.8 },
  { id: "e-9", from: "hidden-3", to: "output", weight: 0.65 },
];

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
  [...edges, { id: "candidate", from, to, weight: 0 }].forEach((edge) => {
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
  const jamIndexRef = useRef(0);
  const idRef = useRef(20);
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
  const [jamOn, setJamOn] = useState(false);
  const [showBoredPrompt, setShowBoredPrompt] = useState(false);
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
    const rectangle = svgRef.current?.getBoundingClientRect();
    if (!rectangle) return { x: 0, y: 0 };
    return {
      x: ((clientX - rectangle.left) / rectangle.width) * 1100,
      y: ((clientY - rectangle.top) / rectangle.height) * 620,
    };
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
    const edge = { id: `edge-${idRef.current}`, from, to, weight: 0.5 };
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

  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!dragging) return;
    const point = pointFromPointer(event.clientX, event.clientY);
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

  const playValues = useCallback(
    (nextValues: Record<string, number>) => {
      const context = audioRef.current;
      if (!context) return;
      const soundingNodes = nodes
        .filter((node) => node.kind !== "input")
        .slice(0, 6);
      const scale = [0, 3, 5, 7, 10, 12];
      const now = context.currentTime;
      soundingNodes.forEach((node, index) => {
        const strength = Math.min(1, Math.abs(nextValues[node.id] ?? 0));
        if (strength < 0.05) return;
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        const semitones = scale[index % scale.length] + ((nextValues[node.id] ?? 0) > 0.55 ? 12 : 0);
        oscillator.type = node.kind === "output" ? "triangle" : "sine";
        oscillator.frequency.value = 146.83 * 2 ** (semitones / 12);
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.012 + strength * 0.022, now + 0.025);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.31);
        oscillator.connect(gain);
        gain.connect(context.destination);
        oscillator.start(now);
        oscillator.stop(now + 0.33);
      });
    },
    [nodes],
  );

  const startJam = async () => {
    if (!audioRef.current) {
      const AudioContextClass =
        window.AudioContext ??
        (window as typeof window & { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (AudioContextClass) audioRef.current = new AudioContextClass();
    }
    if (audioRef.current?.state === "suspended") await audioRef.current.resume();
    setJamOn(true);
    setShowBoredPrompt(false);
    window.sessionStorage.setItem("nnvl-bored-prompt", "used");
    setStatus("Neural Jam is playing. Each pulse turns neuron activity into a short chord.");
  };

  useEffect(() => {
    if (!jamOn) return;
    const tick = () => {
      const sample = samples[jamIndexRef.current % Math.max(1, samples.length)] ?? {
        x: 0,
        y: 0,
        label: 0,
      };
      jamIndexRef.current += 7;
      const nextInputs = sampleInputs(nodes, sample);
      const pass = runGraph(nodes, edges, nextInputs);
      setInputValues(nextInputs);
      setValues(pass.values);
      setPulse((value) => value + 1);
      playValues(pass.values);
    };
    tick();
    const timer = window.setInterval(tick, 560);
    return () => window.clearInterval(timer);
  }, [edges, jamOn, nodes, playValues, samples]);

  useEffect(() => {
    const section = sectionRef.current;
    if (!section || jamOn) return;
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
          }, 30000);
        }
      },
      { threshold: [0.45] },
    );
    observer.observe(section);
    return () => {
      if (timer) clearTimeout(timer);
      observer.disconnect();
    };
  }, [jamOn]);

  useEffect(
    () => () => {
      if (audioRef.current) void audioRef.current.close();
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
          then train or test the graph against the dataset above. Double-click
          empty space to add a hidden neuron.
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
        </div>

        <div className="workbench-body">
          <div className="workbench-canvas-wrap">
            <svg
              ref={svgRef}
              className="workbench-canvas"
              viewBox="0 0 1100 620"
              role="img"
              aria-label="Editable feed-forward neural network. Drag neurons or use the toolbar and inspector to edit the graph."
              onPointerMove={handlePointerMove}
              onPointerUp={() => setDragging(null)}
              onPointerCancel={() => setDragging(null)}
              onPointerDown={() => setSelection(null)}
              onDoubleClick={(event) => {
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
                return (
                  <g key={edge.id}>
                    <line
                      x1={source.x}
                      y1={source.y}
                      x2={target.x}
                      y2={target.y}
                      className={selected ? "work-edge selected" : "work-edge"}
                      stroke={edge.weight >= 0 ? "#176b65" : "#b65a3a"}
                      strokeWidth={1.5 + Math.min(4, Math.abs(edge.weight) * 1.25)}
                      markerEnd="url(#edge-arrow)"
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        setSelection({ kind: "edge", id: edge.id });
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
                <p>Feeling bored?</p>
                <button type="button" onClick={() => void startJam()}>Click here</button>
              </div>
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
                  <small>Petrol carries a positive weight; rust carries a negative one.</small>
                </>
              ) : null}
            </section>

            <section className="jam-panel">
              <p className="eyebrow">Neural Jam</p>
              <strong>Hear the network fire</strong>
              <p>Dataset samples become a short sequence. Pitch and volume follow the current neuron activations.</p>
              <button
                type="button"
                className={jamOn ? "active" : undefined}
                onClick={() => (jamOn ? setJamOn(false) : void startJam())}
              >
                {jamOn ? "Stop Neural Jam" : "Play Neural Jam"}
              </button>
            </section>
          </aside>
        </div>
      </div>
      <p className="workbench-note">
        Lab note: the editor prevents circular connections so its graphs remain trainable feed-forward networks. The output neuron uses a sigmoid; hidden neurons can be changed individually.
      </p>
    </section>
  );
}
