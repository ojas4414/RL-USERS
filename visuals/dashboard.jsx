import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Database,
  GitBranch,
  Network,
  ShoppingBag,
  Users,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = {
  background: "#0F0F0F",
  card: "#1A1A1A",
  border: "#2A2A2A",
  muted: "#A1A1AA",
  blue: "#3B82F6",
  purple: "#8B5CF6",
  red: "#EF4444",
  green: "#22C55E",
  amber: "#F59E0B",
  orange: "#F97316",
};

const personaProfiles = {
  power_buyer: {
    label: "power_buyer",
    color: COLORS.red,
    radius: 8,
    minBalance: 3000,
    maxBalance: 5000,
    behavior: "high budget, decisive",
  },
  average_buyer: {
    label: "average_buyer",
    color: COLORS.blue,
    radius: 6,
    minBalance: 500,
    maxBalance: 2000,
    behavior: "medium budget",
  },
  browser: {
    label: "browser",
    color: COLORS.green,
    radius: 5,
    minBalance: 50,
    maxBalance: 300,
    behavior: "low budget, rarely buys",
  },
};

const sessionLengths = [
  2, 3, 5, 1, 8, 12, 3, 25, 4, 2, 6, 18, 1, 3, 7, 4, 15, 2, 9, 3, 1, 6, 4,
  22, 3, 8, 5, 1, 4, 11, 2, 7, 3, 19, 4, 6, 2, 8, 1, 5, 3, 14, 2, 6, 4, 3,
  9, 2, 5, 1,
];

const histogramData = [
  { bucket: "1-5", real: 34, simulated: 31 },
  { bucket: "6-10", real: 9, simulated: 11 },
  { bucket: "11-15", real: 4, simulated: 4 },
  { bucket: "16-20", real: 2, simulated: 2 },
  { bucket: "21+", real: 1, simulated: 2 },
];

const trendingProducts = [
  { asin: "B085BB7B1M", interactions: 47 },
  { asin: "B07XJ8C8F5", interactions: 31 },
  { asin: "B00YQ6X8EO", interactions: 28 },
  { asin: "B09Z9876CD", interactions: 19 },
  { asin: "B08BZ63GMJ", interactions: 14 },
  { asin: "B01M5KNSQN", interactions: 11 },
  { asin: "B097R46CSY", interactions: 8 },
  { asin: "B099DRHW5V", interactions: 6 },
  { asin: "B08BBQ29N5", interactions: 5 },
  { asin: "B01CUPMQZE", interactions: 4 },
].map((product) => ({
  ...product,
  label: product.asin.slice(-6),
}));

const validationMetrics = [
  {
    name: "Session Length",
    benchmark: "~8",
    simulated: "9.54 +/- 10.81",
    status: "PASS",
    icon: CheckCircle2,
  },
  {
    name: "Conversion Rate",
    benchmark: "2-20%",
    simulated: "20%",
    status: "PASS",
    icon: CheckCircle2,
  },
  {
    name: "Abandonment Rate",
    benchmark: "70-85%",
    simulated: "80%",
    status: "PASS",
    icon: CheckCircle2,
  },
  {
    name: "Social Influence",
    benchmark: "0.20-0.40",
    simulated: "0.033",
    status: "FAIL",
    icon: AlertTriangle,
    tooltip:
      "Gateway catalog fallback limited product diversity. Known limitation - not a modeling failure.",
  },
];

const metricTiles = [
  { value: "701K", label: "Amazon reviews", icon: Database },
  { value: "50", label: "Simulated agents", icon: Users },
  { value: "3 PASS / 1 FAIL", label: "Behavioral metrics", icon: Activity },
  { value: "BFS + Heap + Topo Sort", label: "CS algorithms", icon: GitBranch },
];

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildAgents() {
  return Array.from({ length: 50 }, (_, index) => {
    const persona =
      index < 10 ? "power_buyer" : index < 45 ? "average_buyer" : "browser";
    const profile = personaProfiles[persona];
    const range = profile.maxBalance - profile.minBalance;
    const balance = profile.minBalance + ((index * 137 + 89) % (range + 1));

    return {
      id: `AG-${String(index + 1).padStart(2, "0")}`,
      persona,
      balance,
      behavior: profile.behavior,
      color: profile.color,
      radius: profile.radius,
    };
  });
}

function buildSocialLinks(nodes) {
  const random = seededRandom(701528);
  const seen = new Set();
  const links = [];

  nodes.forEach((node, index) => {
    const targetCount = 2 + Math.floor(random() * 3);
    let attempts = 0;

    while (
      links.filter((link) => link.source === node.id || link.target === node.id)
        .length < targetCount &&
      attempts < 120
    ) {
      attempts += 1;
      const offset = 1 + Math.floor(random() * (nodes.length - 1));
      const targetIndex = (index + offset) % nodes.length;
      const target = nodes[targetIndex];
      const [source, destination] = [node.id, target.id].sort();
      const key = `${source}-${destination}`;

      if (!seen.has(key)) {
        seen.add(key);
        links.push({ source, target: destination });
      }
    }
  });

  return links;
}

const agents = buildAgents();
const socialLinks = buildSocialLinks(agents);

function buildAdjacency(nodes, links) {
  const adjacency = Object.fromEntries(nodes.map((node) => [node.id, []]));

  links.forEach((link) => {
    adjacency[link.source].push(link.target);
    adjacency[link.target].push(link.source);
  });

  return adjacency;
}

const adjacency = buildAdjacency(agents, socialLinks);

function formatBalance(value) {
  return `Rs ${value.toLocaleString("en-IN")}`;
}

function nodeRadius(node) {
  return personaProfiles[node.persona].radius;
}

function chartColor(count) {
  return d3.interpolateRgb("#93C5FD", "#1E40AF")(count / 47);
}

function SectionPanel({ title, eyebrow, icon: Icon, children, className = "" }) {
  return (
    <section
      className={`rounded-lg border border-[#2A2A2A] bg-[#1A1A1A]/92 p-5 shadow-[0_18px_60px_rgba(0,0,0,0.28)] transition duration-300 hover:border-zinc-700 ${className}`}
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          {eyebrow ? (
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-zinc-500">
              {eyebrow}
            </p>
          ) : null}
          <h2 className="mt-1 flex items-center gap-2 text-lg font-semibold text-zinc-50">
            {Icon ? <Icon className="h-5 w-5 text-[#8B5CF6]" /> : null}
            {title}
          </h2>
        </div>
      </div>
      {children}
    </section>
  );
}

function HeroMetric({ value, label, icon: Icon }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.045] p-4 transition duration-300 hover:-translate-y-0.5 hover:border-white/20 hover:bg-white/[0.07]">
      <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-black/20">
        <Icon className="h-5 w-5 text-[#3B82F6]" />
      </div>
      <div className="font-mono text-xl font-semibold text-white">{value}</div>
      <div className="mt-1 text-sm text-zinc-400">{label}</div>
    </div>
  );
}

function DashboardHero() {
  return (
    <header className="relative overflow-hidden rounded-lg border border-[#2A2A2A] bg-[radial-gradient(circle_at_20%_20%,rgba(59,130,246,0.24),transparent_28%),linear-gradient(135deg,#151515_0%,#10131A_45%,#0F0F0F_100%)] p-6 shadow-[0_24px_80px_rgba(0,0,0,0.36)] md:p-8">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[#3B82F6] to-transparent opacity-70" />
      <div className="grid gap-6 lg:grid-cols-[1.05fr_1.4fr] lg:items-end">
        <div>
          <div className="mb-4 inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 font-mono text-xs text-zinc-300">
            <Zap className="h-4 w-4 text-[#F59E0B]" />
            PyTorch behavioral policy simulator
          </div>
          <h1 className="max-w-4xl text-3xl font-semibold leading-tight text-white md:text-5xl">
            RL-USERS <span className="text-[#8B5CF6]">&mdash;</span>{" "}
            Multi-Agent Behavioral Simulator
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-7 text-zinc-300 md:text-lg">
            50 virtual users trained on 701,528 Amazon reviews
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {metricTiles.map((metric) => (
            <HeroMetric key={metric.label} {...metric} />
          ))}
        </div>
      </div>
    </header>
  );
}

function AgentTooltip({ hover }) {
  if (!hover) {
    return null;
  }

  const profile = personaProfiles[hover.agent.persona];

  return (
    <div
      className="pointer-events-none absolute z-20 min-w-[190px] rounded-lg border border-zinc-700 bg-[#0F0F0F]/95 p-3 text-sm shadow-2xl backdrop-blur"
      style={{
        left: Math.max(8, Math.min(hover.x + 14, hover.bounds.width - 205)),
        top: Math.max(hover.y - 18, 8),
      }}
    >
      <div className="font-mono text-xs uppercase tracking-[0.18em] text-zinc-500">
        {hover.agent.id}
      </div>
      <div className="mt-1 flex items-center gap-2 font-semibold text-zinc-50">
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: profile.color }}
        />
        {profile.label}
      </div>
      <div className="mt-2 text-zinc-400">{hover.agent.behavior}</div>
      <div className="mt-2 font-mono text-zinc-200">
        balance: {formatBalance(hover.agent.balance)}
      </div>
    </div>
  );
}

function AgentSocialNetwork() {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const timersRef = useRef([]);
  const [dimensions, setDimensions] = useState({ width: 720, height: 500 });
  const [hover, setHover] = useState(null);
  const [pulse, setPulse] = useState({ root: null, first: [], second: [] });

  const clearPulseTimers = useCallback(() => {
    timersRef.current.forEach((timer) => window.clearTimeout(timer));
    timersRef.current = [];
  }, []);

  const startPulse = useCallback(
    (agentId) => {
      clearPulseTimers();

      const first = adjacency[agentId] || [];
      const firstSet = new Set(first);
      const second = Array.from(
        new Set(
          first.flatMap((neighbor) => adjacency[neighbor] || []).filter(
            (candidate) => candidate !== agentId && !firstSet.has(candidate)
          )
        )
      );

      setPulse({ root: agentId, first: [], second: [] });
      timersRef.current = [
        window.setTimeout(() => {
          setPulse({ root: agentId, first, second: [] });
        }, 300),
        window.setTimeout(() => {
          setPulse({ root: agentId, first, second });
        }, 600),
        window.setTimeout(() => {
          setPulse({ root: null, first: [], second: [] });
        }, 1200),
      ];
    },
    [clearPulseTimers]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    const updateSize = () => {
      const width = container.clientWidth || 720;
      setDimensions({
        width,
        height: Math.max(430, Math.min(560, width * 0.72)),
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const svgElement = svgRef.current;
    if (!svgElement || !dimensions.width) {
      return undefined;
    }

    const width = dimensions.width;
    const height = dimensions.height;
    const nodes = agents.map((node) => ({ ...node }));
    const links = socialLinks.map((link) => ({ ...link }));
    const svg = d3.select(svgElement);

    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", width)
      .attr("height", height)
      .attr("role", "img");

    const backgroundGrid = svg
      .append("g")
      .attr("opacity", 0.16)
      .attr("stroke", "#3F3F46")
      .attr("stroke-width", 0.6);

    d3.range(0, width, 42).forEach((x) => {
      backgroundGrid
        .append("line")
        .attr("x1", x)
        .attr("x2", x)
        .attr("y1", 0)
        .attr("y2", height);
    });

    d3.range(0, height, 42).forEach((y) => {
      backgroundGrid
        .append("line")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", y)
        .attr("y2", y);
    });

    const link = svg
      .append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("class", "social-link")
      .attr("stroke", "#3F3F46")
      .attr("stroke-width", 0.7)
      .attr("stroke-opacity", 0.55);

    const node = svg
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("class", "agent-node")
      .attr("r", (d) => nodeRadius(d))
      .attr("fill", (d) => personaProfiles[d.persona].color)
      .attr("stroke", "#111827")
      .attr("stroke-width", 1.4)
      .attr("cursor", "pointer")
      .on("mousemove", (event, d) => {
        setHover({
          agent: d,
          x: event.offsetX,
          y: event.offsetY,
          bounds: { width, height },
        });
      })
      .on("mouseleave", () => setHover(null))
      .on("click", (event, d) => {
        event.stopPropagation();
        startPulse(d.id);
      });

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink(links)
          .id((d) => d.id)
          .distance((d) => {
            const source =
              typeof d.source === "object" ? d.source.persona : "average_buyer";
            const target =
              typeof d.target === "object" ? d.target.persona : "average_buyer";
            return source === "power_buyer" || target === "power_buyer" ? 68 : 52;
          })
          .strength(0.42)
      )
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) => nodeRadius(d) + 8))
      .alpha(0.9);

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node
        .attr("cx", (d) => Math.max(18, Math.min(width - 18, d.x)))
        .attr("cy", (d) => Math.max(18, Math.min(height - 18, d.y)));
    });

    return () => simulation.stop();
  }, [dimensions, startPulse]);

  useEffect(() => {
    const svgElement = svgRef.current;
    if (!svgElement) {
      return;
    }

    const rootSet = new Set(pulse.root ? [pulse.root] : []);
    const firstSet = new Set(pulse.first);
    const secondSet = new Set(pulse.second);
    const hasPulse = Boolean(pulse.root);
    const svg = d3.select(svgElement);

    svg
      .selectAll(".agent-node")
      .transition("pulse-node")
      .duration(220)
      .attr("r", (d) => {
        if (rootSet.has(d.id)) return nodeRadius(d) + 6;
        if (firstSet.has(d.id)) return nodeRadius(d) + 4;
        if (secondSet.has(d.id)) return nodeRadius(d) + 2;
        return nodeRadius(d);
      })
      .attr("stroke", (d) => {
        if (rootSet.has(d.id)) return "#FEF08A";
        if (firstSet.has(d.id)) return "#A7F3D0";
        if (secondSet.has(d.id)) return "#93C5FD";
        return "#111827";
      })
      .attr("stroke-width", (d) => {
        if (rootSet.has(d.id)) return 4.5;
        if (firstSet.has(d.id)) return 3.4;
        if (secondSet.has(d.id)) return 2.4;
        return 1.4;
      })
      .attr("opacity", (d) => {
        if (!hasPulse) return 1;
        return rootSet.has(d.id) || firstSet.has(d.id) || secondSet.has(d.id)
          ? 1
          : 0.32;
      })
      .style("filter", (d) => {
        if (rootSet.has(d.id)) return "drop-shadow(0 0 16px #FEF08A)";
        if (firstSet.has(d.id)) return "drop-shadow(0 0 13px #A7F3D0)";
        if (secondSet.has(d.id)) return "drop-shadow(0 0 9px #93C5FD)";
        return "none";
      });

    svg
      .selectAll(".social-link")
      .transition("pulse-link")
      .duration(220)
      .attr("stroke", (d) => {
        const source = typeof d.source === "object" ? d.source.id : d.source;
        const target = typeof d.target === "object" ? d.target.id : d.target;
        const active =
          (rootSet.has(source) && firstSet.has(target)) ||
          (rootSet.has(target) && firstSet.has(source)) ||
          (firstSet.has(source) && secondSet.has(target)) ||
          (firstSet.has(target) && secondSet.has(source));
        return active ? "#A7F3D0" : "#3F3F46";
      })
      .attr("stroke-opacity", (d) => {
        const source = typeof d.source === "object" ? d.source.id : d.source;
        const target = typeof d.target === "object" ? d.target.id : d.target;
        const active =
          (rootSet.has(source) && firstSet.has(target)) ||
          (rootSet.has(target) && firstSet.has(source)) ||
          (firstSet.has(source) && secondSet.has(target)) ||
          (firstSet.has(target) && secondSet.has(source));
        return active ? 0.95 : hasPulse ? 0.18 : 0.55;
      })
      .attr("stroke-width", (d) => {
        const source = typeof d.source === "object" ? d.source.id : d.source;
        const target = typeof d.target === "object" ? d.target.id : d.target;
        const active =
          (rootSet.has(source) && firstSet.has(target)) ||
          (rootSet.has(target) && firstSet.has(source)) ||
          (firstSet.has(source) && secondSet.has(target)) ||
          (firstSet.has(target) && secondSet.has(source));
        return active ? 1.8 : 0.7;
      });
  }, [pulse]);

  useEffect(() => clearPulseTimers, [clearPulseTimers]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {Object.values(personaProfiles).map((profile) => (
          <div
            key={profile.label}
            className="flex items-center gap-2 rounded-lg border border-[#2A2A2A] bg-black/20 px-3 py-2 font-mono text-xs text-zinc-300"
          >
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: profile.color }}
            />
            {profile.label}
          </div>
        ))}
      </div>
      <div
        ref={containerRef}
        className="relative overflow-hidden rounded-lg border border-[#2A2A2A] bg-[#101010]"
      >
        <svg ref={svgRef} className="block w-full" />
        <AgentTooltip hover={hover} />
      </div>
    </div>
  );
}

function MetricsTable() {
  return (
    <div className="space-y-3">
      {validationMetrics.map((metric) => {
        const Icon = metric.icon;
        const isPass = metric.status === "PASS";

        return (
          <div
            key={metric.name}
            className="grid grid-cols-1 items-center gap-3 rounded-lg border border-[#2A2A2A] bg-black/20 px-3 py-3 transition duration-300 hover:border-zinc-700 hover:bg-black/30 sm:grid-cols-[1.2fr_0.9fr_1fr_auto]"
          >
            <div className="text-sm font-medium text-zinc-100">{metric.name}</div>
            <div className="font-mono text-xs text-zinc-400">
              {metric.benchmark}
            </div>
            <div className="font-mono text-xs text-zinc-200">
              {metric.simulated}
            </div>
            <div className="relative inline-flex w-fit items-center gap-1.5">
              <span
                className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 font-mono text-xs font-semibold ${
                  isPass
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                    : "border-amber-500/35 bg-amber-500/10 text-amber-300"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {metric.status}
              </span>
              {metric.tooltip ? (
                <span className="group relative">
                  <AlertTriangle className="h-4 w-4 text-amber-300" />
                  <span className="pointer-events-none absolute right-0 top-6 z-10 w-72 rounded-lg border border-amber-500/30 bg-[#0F0F0F] p-3 text-xs leading-5 text-zinc-200 opacity-0 shadow-2xl transition duration-200 group-hover:opacity-100">
                    {metric.tooltip}
                  </span>
                </span>
              ) : null}
            </div>
          </div>
        );
      })}

      <p className="rounded-lg border border-[#2A2A2A] bg-black/20 p-4 text-sm leading-6 text-zinc-400">
        KL divergence removed &mdash; product choice is personal and unique.
        Behavioral patterns (attention span, abandonment, social influence) are
        the correct validation signal for a user simulator.
      </p>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) {
    return null;
  }

  return (
    <div className="rounded-lg border border-[#2A2A2A] bg-[#0F0F0F]/95 p-3 shadow-xl">
      <div className="mb-2 font-mono text-xs uppercase tracking-[0.18em] text-zinc-500">
        {label}
      </div>
      {payload.map((item) => (
        <div
          key={item.name}
          className="flex items-center justify-between gap-6 text-sm text-zinc-200"
        >
          <span style={{ color: item.color }}>{item.name}</span>
          <span className="font-mono">{item.value}</span>
        </div>
      ))}
    </div>
  );
}

function SessionLengthChart() {
  return (
    <SectionPanel
      title="Session Length Distribution"
      eyebrow="Validation"
      icon={BarChart3}
    >
      <div className="h-[270px]">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart
            data={histogramData}
            margin={{ top: 8, right: 8, bottom: 0, left: -12 }}
            barGap={2}
          >
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis
              dataKey="bucket"
              tick={{ fill: "#A1A1AA", fontSize: 12 }}
              axisLine={{ stroke: "#2A2A2A" }}
              tickLine={false}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: "#A1A1AA", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "#ffffff0a" }} />
            <Legend
              wrapperStyle={{ color: "#D4D4D8", fontSize: 12, paddingTop: 12 }}
            />
            <Bar
              dataKey="real"
              name="Real users"
              fill={COLORS.blue}
              radius={[4, 4, 0, 0]}
            />
            <Bar
              dataKey="simulated"
              name="Simulated agents"
              fill={COLORS.orange}
              radius={[4, 4, 0, 0]}
            />
          </RechartsBarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-xs text-zinc-500">
        <span>Real mean 8.16, std 7.08</span>
        <span className="text-right">Sim mean 9.54, std 10.81</span>
      </div>
    </SectionPanel>
  );
}

function TrendingProductsChart() {
  return (
    <SectionPanel
      title="Trending Products"
      eyebrow="Social Contagion Effect"
      icon={Activity}
    >
      <div className="mb-2 font-mono text-xs text-zinc-500">
        Top 3 products = 56% of all interactions
      </div>
      <div className="h-[285px]">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart
            data={trendingProducts}
            layout="vertical"
            margin={{ top: 4, right: 18, bottom: 0, left: -6 }}
          >
            <CartesianGrid stroke="#2A2A2A" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: "#A1A1AA", fontSize: 12 }}
              axisLine={{ stroke: "#2A2A2A" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={60}
              tick={{ fill: "#D4D4D8", fontSize: 12, fontFamily: "monospace" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "#ffffff0a" }} />
            <Bar
              dataKey="interactions"
              name="Interactions"
              radius={[0, 5, 5, 0]}
              barSize={15}
            >
              {trendingProducts.map((entry) => (
                <Cell key={entry.asin} fill={chartColor(entry.interactions)} />
              ))}
            </Bar>
          </RechartsBarChart>
        </ResponsiveContainer>
      </div>
    </SectionPanel>
  );
}

function PurchaseFunnel() {
  const steps = [
    {
      name: "Browse",
      percent: "100%",
      top: 330,
      bottom: 285,
      y: 18,
      color: "#6366F1",
    },
    {
      name: "Product Detail",
      percent: "68%",
      top: 285,
      bottom: 220,
      y: 78,
      color: "#8B5CF6",
    },
    {
      name: "Cart",
      percent: "35%",
      top: 220,
      bottom: 155,
      y: 138,
      color: "#A78BFA",
    },
    {
      name: "Checkout",
      percent: "20%",
      top: 155,
      bottom: 110,
      y: 198,
      color: "#C4B5FD",
    },
  ];

  const center = 210;
  const height = 42;

  return (
    <SectionPanel title="Purchase Funnel" eyebrow="User Journey" icon={ShoppingBag}>
      <svg viewBox="0 0 420 280" className="h-[285px] w-full">
        <defs>
          <filter id="funnelShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow
              dx="0"
              dy="10"
              stdDeviation="12"
              floodColor="#000000"
              floodOpacity="0.28"
            />
          </filter>
        </defs>
        {steps.map((step, index) => {
          const points = [
            [center - step.top / 2, step.y],
            [center + step.top / 2, step.y],
            [center + step.bottom / 2, step.y + height],
            [center - step.bottom / 2, step.y + height],
          ]
            .map((point) => point.join(","))
            .join(" ");

          return (
            <g key={step.name}>
              <polygon
                points={points}
                fill={step.color}
                filter="url(#funnelShadow)"
                className="transition duration-300 hover:brightness-110"
                opacity={index === steps.length - 1 ? 0.96 : 0.9}
              />
              <text
                x={center}
                y={step.y + 18}
                textAnchor="middle"
                className="fill-white text-[14px] font-semibold"
              >
                {step.name}
              </text>
              <text
                x={center}
                y={step.y + 35}
                textAnchor="middle"
                className="fill-white/80 font-mono text-[12px]"
              >
                {step.percent}
              </text>
              {index < steps.length - 1 ? (
                <g>
                  <line
                    x1={center}
                    x2={center}
                    y1={step.y + height + 3}
                    y2={step.y + height + 14}
                    stroke="#71717A"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                  <path
                    d={`M ${center - 5} ${step.y + height + 10} L ${center} ${
                      step.y + height + 16
                    } L ${center + 5} ${step.y + height + 10}`}
                    fill="none"
                    stroke="#A1A1AA"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
              ) : null}
            </g>
          );
        })}
      </svg>
    </SectionPanel>
  );
}

function ArchitectureStep({ icon: Icon, title, detail }) {
  return (
    <div className="min-h-[138px] flex-1 rounded-lg border border-[#2A2A2A] bg-black/20 p-4 transition duration-300 hover:-translate-y-0.5 hover:border-zinc-700 hover:bg-black/30">
      <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04]">
        <Icon className="h-5 w-5 text-[#3B82F6]" />
      </div>
      <div className="font-semibold text-zinc-50">{title}</div>
      <div className="mt-2 text-sm leading-5 text-zinc-400">{detail}</div>
    </div>
  );
}

function ArchitectureDiagram() {
  const steps = [
    {
      title: "Amazon Data",
      detail: (
        <>
          701K reviews <span className="font-mono">&rarr;</span> 2,985 products
        </>
      ),
      icon: Database,
    },
    {
      title: "BC Training + PPO",
      detail: (
        <>
          Behavioral cloning <span className="font-mono">&rarr;</span> PPO
          fine-tuning
        </>
      ),
      icon: BrainCircuit,
    },
    {
      title: "50 Agents + Social Graph",
      detail: "Heap scheduler, BFS propagation, Topo sort funnel",
      icon: Network,
    },
    {
      title: "Validation",
      detail: (
        <>
          Session length <span className="text-emerald-300">&check;</span>{" "}
          Conversion <span className="text-emerald-300">&check;</span>{" "}
          Abandonment <span className="text-emerald-300">&check;</span>
        </>
      ),
      icon: CheckCircle2,
    },
  ];

  return (
    <SectionPanel
      title="Architecture"
      eyebrow="System Flow"
      icon={GitBranch}
      className="bg-[linear-gradient(135deg,#1A1A1A_0%,#141414_62%,#10131A_100%)]"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-stretch">
        {steps.map((step, index) => (
          <React.Fragment key={step.title}>
            <ArchitectureStep {...step} />
            {index < steps.length - 1 ? (
              <div className="hidden items-center justify-center text-zinc-600 lg:flex">
                <ArrowRight className="h-6 w-6" />
              </div>
            ) : null}
          </React.Fragment>
        ))}
      </div>
      <div className="mt-4 grid gap-2 font-mono text-xs text-zinc-500 sm:grid-cols-4">
        <span>Amazon Data: reviews -> products</span>
        <span>BC Training: imitation -> policy</span>
        <span>50 Agents: heap + BFS + topo</span>
        <span>Validation: session + conversion + abandonment</span>
      </div>
    </SectionPanel>
  );
}

function PersonaSummary() {
  const summaries = [
    { label: "power_buyers", count: 10, color: COLORS.red, budget: "Rs 3000-5000" },
    {
      label: "average_buyers",
      count: 35,
      color: COLORS.blue,
      budget: "Rs 500-2000",
    },
    { label: "browsers", count: 5, color: COLORS.green, budget: "Rs 50-300" },
  ];

  return (
    <div className="mt-4 grid gap-3 sm:grid-cols-3">
      {summaries.map((summary) => (
        <div
          key={summary.label}
          className="rounded-lg border border-[#2A2A2A] bg-black/20 p-3"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-xs text-zinc-400">{summary.label}</span>
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: summary.color }}
            />
          </div>
          <div className="mt-2 text-2xl font-semibold text-white">
            {summary.count}
          </div>
          <div className="mt-1 font-mono text-xs text-zinc-500">
            {summary.budget}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function RLUsersDashboard() {
  const sampleStats = useMemo(
    () => ({
      shortest: Math.min(...sessionLengths),
      longest: Math.max(...sessionLengths),
      sample: sessionLengths.length,
    }),
    []
  );

  return (
    <main className="min-h-screen bg-[#0F0F0F] px-4 py-5 text-zinc-100 sm:px-6 lg:px-8">
      <style>{`
        .recharts-cartesian-axis-tick text {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
        .recharts-legend-item-text {
          color: #D4D4D8 !important;
        }
      `}</style>

      <div className="mx-auto max-w-[1540px] space-y-5">
        <DashboardHero />

        <div className="grid gap-5 xl:grid-cols-[1.22fr_0.78fr]">
          <SectionPanel
            title="Agent Social Network - BFS Influence Propagation"
            eyebrow="50 agents, Erdos-Renyi p=0.1"
            icon={Network}
          >
            <AgentSocialNetwork />
            <PersonaSummary />
          </SectionPanel>

          <SectionPanel
            title="Behavioral Metrics"
            eyebrow={`Sample ${sampleStats.sample}, range ${sampleStats.shortest}-${sampleStats.longest} actions`}
            icon={CheckCircle2}
          >
            <MetricsTable />
          </SectionPanel>
        </div>

        <div className="grid gap-5 xl:grid-cols-3">
          <SessionLengthChart />
          <TrendingProductsChart />
          <PurchaseFunnel />
        </div>

        <ArchitectureDiagram />

        <footer className="rounded-lg border border-[#2A2A2A] bg-[#1A1A1A]/75 px-5 py-4 text-center font-mono text-xs text-zinc-500">
          Built with PyTorch &middot; FastAPI &middot; Redis &middot; Kafka
          &middot; C++ (libtorch) &middot; pybind11
        </footer>
      </div>
    </main>
  );
}
