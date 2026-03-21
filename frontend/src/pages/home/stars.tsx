import React, { useMemo, useState, useCallback, useRef, useEffect, useLayoutEffect } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { Html, OrbitControls, QuadraticBezierLine } from '@react-three/drei';
import * as d3 from 'd3';
import type { SimulationNodeDatum } from 'd3';
import { Link } from 'umi';
import {
  baguNodes,
  baguLinks,
  BAGU_GROUP_COLOR,
  BAGU_GROUP_LABEL,
  nodeDegrees,
  type BaguNode,
  type BaguGroup,
} from './llmGraphData';

type TabKey = 'all' | BaguGroup;

interface SimNode extends BaguNode, SimulationNodeDatum {}

const sn = (d: SimulationNodeDatum) => d as SimNode;

function runLayout(nodes: BaguNode[], links: { source: string; target: string }[]) {
  const simNodes: SimNode[] = nodes.map((n) => ({ ...n }));
  const simLinks = links.map((l) => ({ ...l }));
  const link = d3
    .forceLink<SimNode, { source: string; target: string }>(simLinks)
    .id((d) => d.id)
    .distance((d) => {
      const s = typeof d.source === 'string' ? simNodes.find((n) => n.id === d.source)! : d.source;
      const t = typeof d.target === 'string' ? simNodes.find((n) => n.id === d.target)! : d.target;
      const ts = (s.tier + t.tier) / 2;
      return 28 + (4 - ts) * 14;
    })
    .strength(0.55);

  const sim = d3
    .forceSimulation(simNodes as SimulationNodeDatum[])
    .force('link', link)
    .force('charge', d3.forceManyBody().strength(-220))
    .force('center', d3.forceCenter(0, 0))
    .force(
      'radial',
      d3
        .forceRadial(
          (d) => (sn(d).tier === 3 ? 0 : sn(d).tier === 2 ? 120 : 200),
          0,
          0,
        )
        .strength((d) => (sn(d).tier === 3 ? 0.18 : 0.06)),
    )
    .force('collision', d3.forceCollide().radius((d) => 14 + (4 - sn(d).tier) * 6))
    .alphaDecay(0.02)
    .velocityDecay(0.85);

  for (let i = 0; i < 420; i++) sim.tick();

  const pos = new Map<string, THREE.Vector3>();
  const zSeed = (s: string) => {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    return ((h % 1000) / 1000 - 0.5) * 3.2;
  };
  simNodes.forEach((n) => {
    const scale = 0.041;
    const x = (n.x ?? 0) * scale;
    const y = -(n.y ?? 0) * scale;
    const z = zSeed(n.id) + (3 - n.tier) * 0.35;
    pos.set(n.id, new THREE.Vector3(x, y, z));
  });
  return pos;
}

function clonePositionMap(src: Map<string, THREE.Vector3>) {
  const next = new Map<string, THREE.Vector3>();
  src.forEach((v, k) => next.set(k, v.clone()));
  return next;
}

/** 二次贝塞尔控制点：向屏幕方向弓起，边 key 决定左右摆向，避免全部同向 */
function bezierMid(
  a: THREE.Vector3,
  b: THREE.Vector3,
  cam: THREE.Vector3,
  edgeKey: string,
): THREE.Vector3 {
  const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
  const chord = new THREE.Vector3().subVectors(b, a);
  const len = chord.length();
  if (len < 1e-6) return mid.clone();
  const view = new THREE.Vector3().subVectors(cam, mid);
  const binorm = new THREE.Vector3().crossVectors(chord, view);
  if (binorm.lengthSq() < 1e-8) binorm.set(0, 1, 0);
  else binorm.normalize();
  let h = 0;
  for (let i = 0; i < edgeKey.length; i++) h = (h * 31 + edgeKey.charCodeAt(i)) | 0;
  const sign = h % 2 === 0 ? 1 : -1;
  const bow = len * 0.24 * sign;
  return mid.addScaledVector(binorm, bow);
}

function StarBackdrop() {
  const ref = useRef<THREE.Points>(null);
  const [positions] = useState(() => {
    const n = 2200;
    const a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const r = 18 + Math.random() * 42;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      a[i * 3] = r * Math.sin(ph) * Math.cos(th);
      a[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
      a[i * 3 + 2] = r * Math.cos(ph);
    }
    return a;
  });
  useFrame((s) => {
    if (ref.current) ref.current.rotation.y = s.clock.elapsedTime * 0.012;
  });
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.055}
        color="#b8d4f8"
        transparent
        opacity={0.42}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/** 曲线星轨：宽而淡的底光 + 细芯线，略像参考图里的丝状光晕叠层 */
function BaguNebulaEdge({
  start,
  end,
  edgeKey,
  color,
  opacity,
  active,
}: {
  start: THREE.Vector3;
  end: THREE.Vector3;
  edgeKey: string;
  color: string;
  opacity: number;
  active: boolean;
}) {
  const { camera } = useThree();
  const mid = useMemo(
    () => bezierMid(start, end, camera.position, edgeKey),
    [start, end, edgeKey, start.x, start.y, start.z, end.x, end.y, end.z, camera.position.x, camera.position.y, camera.position.z],
  );
  const seg = active ? 28 : 22;
  const wideOp = active ? Math.min(0.22, opacity + 0.12) : opacity * 0.55;
  const coreOp = active ? Math.min(0.75, opacity + 0.28) : opacity;

  return (
    <group renderOrder={-3}>
      <QuadraticBezierLine
        start={start}
        end={end}
        mid={mid}
        segments={seg}
        color={color}
        lineWidth={active ? 4 : 3}
        transparent
        opacity={wideOp}
        depthWrite={false}
        toneMapped={false}
      />
      <QuadraticBezierLine
        start={start}
        end={end}
        mid={mid}
        segments={seg}
        color={active ? '#f8fafc' : color}
        lineWidth={active ? 1.4 : 0.85}
        transparent
        opacity={coreOp}
        depthWrite={false}
        toneMapped={false}
      />
    </group>
  );
}

function BaguStarNode({
  node,
  position,
  degree,
  emphasis,
  selected,
  hovered,
  dragging,
  onHover,
  onPointerDown,
}: {
  node: BaguNode;
  position: THREE.Vector3;
  degree: number;
  emphasis: boolean;
  selected: boolean;
  hovered: boolean;
  dragging: boolean;
  onHover: (v: boolean) => void;
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void;
}) {
  const mesh = useRef<THREE.Mesh>(null);
  const glow = useRef<THREE.Mesh>(null);
  const nebula = useRef<THREE.Mesh>(null);
  const color = BAGU_GROUP_COLOR[node.group];
  const degBoost = Math.min(0.18, Math.sqrt(Math.max(1, degree)) * 0.045);
  const base =
    ((node.tier === 3 ? 0.42 : node.tier === 2 ? 0.28 : 0.16) + degBoost) * 0.82;
  const dim = emphasis ? 1 : 0.32;
  const labelSize = node.tier === 3 ? 15 : node.tier === 2 ? 12 : 10;
  const labelOpacity = node.tier >= 2 || selected || hovered ? 1 : 0.42;

  useFrame((s) => {
    const t = s.clock.elapsedTime;
    if (mesh.current) {
      const pulse = selected ? 1 + Math.sin(t * 3) * 0.06 : hovered ? 1.05 : 1;
      mesh.current.scale.setScalar(pulse);
    }
    if (glow.current) {
      const g = dragging ? 1.45 : selected ? 1.55 : hovered ? 1.35 : 1.12;
      glow.current.scale.setScalar(g);
    }
    if (nebula.current) {
      nebula.current.scale.setScalar(1 + Math.sin(t * 0.7 + position.x) * 0.04);
    }
  });

  return (
    <group position={position}>
      <mesh ref={nebula} renderOrder={-1}>
        <sphereGeometry args={[base * 3.4, 14, 14]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.05 * dim + (selected ? 0.06 : 0)}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <mesh ref={glow} renderOrder={1}>
        <sphereGeometry args={[base * 2.2, 20, 20]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.2 * dim + (selected ? 0.28 : 0)}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <mesh
        ref={mesh}
        renderOrder={2}
        onPointerDown={(e) => {
          e.stopPropagation();
          onPointerDown(e);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(true);
        }}
        onPointerOut={() => onHover(false)}
      >
        <sphereGeometry args={[base, 20, 20]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.42 * dim + (selected ? 0.62 : 0) + (hovered ? 0.22 : 0)}
          roughness={0.22}
          metalness={0.5}
          transparent
          opacity={0.28 + 0.72 * dim}
        />
      </mesh>
      <mesh renderOrder={3} scale={selected ? 0.52 : 0.44}>
        <sphereGeometry args={[base, 14, 14]} />
        <meshBasicMaterial
          color="#fffaf0"
          transparent
          opacity={0.22 * dim + (selected ? 0.35 : 0) + (hovered ? 0.12 : 0)}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <Html
        distanceFactor={14}
        position={[0, base + 0.38, 0]}
        center
        style={{ pointerEvents: 'none' }}
        occlude={false}
      >
        <div
          style={{
            padding: '3px 10px',
            borderRadius: 999,
            fontSize: labelSize,
            fontWeight: 600,
            color: '#f8fafc',
            textShadow: '0 0 10px rgba(0,0,0,0.92), 0 0 2px #000',
            whiteSpace: 'nowrap',
            opacity: (0.35 + 0.65 * dim + (selected ? 0.25 : 0)) * labelOpacity,
            WebkitFontSmoothing: 'subpixel-antialiased',
            MozOsxFontSmoothing: 'auto',
            textRendering: 'geometricPrecision',
            filter: 'none',
            transform: 'translateZ(0)',
          }}
        >
          {node.label}
        </div>
      </Html>
    </group>
  );
}

const EDGE_DIM = '#9ca3af';
const DRAG_THRESHOLD_PX = 6;

type DragSession = {
  id: string;
  plane: THREE.Plane;
  grab: THREE.Vector3;
  pointerId: number;
  startX: number;
  startY: number;
  moved: boolean;
};

function Scene({
  positions,
  setPositions,
  degrees,
  tab,
  selectedId,
  hoveredId,
  draggingId,
  onHover,
  onTapNode,
  onDragState,
}: {
  positions: Map<string, THREE.Vector3>;
  setPositions: React.Dispatch<React.SetStateAction<Map<string, THREE.Vector3>>>;
  degrees: Map<string, number>;
  tab: TabKey;
  selectedId: string | null;
  hoveredId: string | null;
  draggingId: string | null;
  onHover: (id: string | null) => void;
  onTapNode: (id: string) => void;
  onDragState: (id: string | null) => void;
}) {
  /** drei OrbitControls 实例，仅需 enabled */
  const orbitRef = useRef<{ enabled: boolean } | null>(null);
  const dragRef = useRef<DragSession | null>(null);
  const { camera, gl, raycaster } = useThree();

  useLayoutEffect(() => {
    gl.toneMapping = THREE.ACESFilmicToneMapping;
    gl.toneMappingExposure = 1.06;
  }, [gl]);
  const ndc = useRef(new THREE.Vector2());
  const hit = useRef(new THREE.Vector3());

  const emphasis = useCallback(
    (n: BaguNode) => tab === 'all' || n.group === tab,
    [tab],
  );

  useEffect(() => {
    const dom = gl.domElement;

    const onMove = (e: PointerEvent) => {
      const s = dragRef.current;
      if (!s || e.pointerId !== s.pointerId) return;
      const dx = e.clientX - s.startX;
      const dy = e.clientY - s.startY;
      if (!s.moved && dx * dx + dy * dy > DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) {
        s.moved = true;
        onDragState(s.id);
        if (orbitRef.current) orbitRef.current.enabled = false;
        dom.style.cursor = 'grabbing';
      }
      if (!s.moved) return;

      const rect = dom.getBoundingClientRect();
      ndc.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc.current, camera);
      if (raycaster.ray.intersectPlane(s.plane, hit.current)) {
        const nextPos = hit.current.clone().sub(s.grab);
        setPositions((prev) => {
          const m = new Map(prev);
          m.set(s.id, nextPos);
          return m;
        });
      }
    };

    const onUp = (e: PointerEvent) => {
      const s = dragRef.current;
      if (!s || e.pointerId !== s.pointerId) return;
      try {
        dom.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      if (!s.moved) onTapNode(s.id);
      dragRef.current = null;
      onDragState(null);
      dom.style.cursor = '';
      if (orbitRef.current) orbitRef.current.enabled = true;
    };

    dom.addEventListener('pointermove', onMove);
    dom.addEventListener('pointerup', onUp);
    dom.addEventListener('pointercancel', onUp);
    return () => {
      dom.removeEventListener('pointermove', onMove);
      dom.removeEventListener('pointerup', onUp);
      dom.removeEventListener('pointercancel', onUp);
    };
  }, [camera, gl, raycaster, onTapNode, onDragState, setPositions]);

  const onNodePointerDown = useCallback(
    (nodeId: string, worldHit: THREE.Vector3, e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();
      const nodePos = positions.get(nodeId);
      if (!nodePos) return;
      const normal = new THREE.Vector3();
      camera.getWorldDirection(normal);
      normal.negate();
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, worldHit);
      const grab = worldHit.clone().sub(nodePos);
      dragRef.current = {
        id: nodeId,
        plane,
        grab,
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        moved: false,
      };
      try {
        gl.domElement.setPointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    },
    [camera, gl, positions],
  );

  return (
    <>
      <color attach="background" args={['#020617']} />
      <fog attach="fog" args={['#020617', 18, 62]} />
      <ambientLight intensity={0.42} />
      <pointLight position={[12, 8, 10]} intensity={1.05} color="#e2e8f0" />
      <pointLight position={[-10, -6, 8]} intensity={0.4} color="#38bdf8" />
      <StarBackdrop />
      {baguLinks.map((l) => {
        const a = positions.get(l.source);
        const b = positions.get(l.target);
        if (!a || !b) return null;
        const na = baguNodes.find((x) => x.id === l.source);
        const nb = baguNodes.find((x) => x.id === l.target);
        const em = na && nb && emphasis(na) && emphasis(nb);
        const active = !!(
          (selectedId && (l.source === selectedId || l.target === selectedId)) ||
          (hoveredId && (l.source === hoveredId || l.target === hoveredId)) ||
          (draggingId && (l.source === draggingId || l.target === draggingId))
        );
        const colTint = na ? BAGU_GROUP_COLOR[na.group] : EDGE_DIM;
        const col = active ? colTint : EDGE_DIM;
        const op = active ? 0.5 : em ? 0.13 : 0.038;
        return (
          <BaguNebulaEdge
            key={`${l.source}-${l.target}`}
            start={a}
            end={b}
            edgeKey={`${l.source}|${l.target}`}
            color={col}
            opacity={op}
            active={active}
          />
        );
      })}
      {baguNodes.map((node) => {
        const p = positions.get(node.id);
        if (!p) return null;
        return (
          <BaguStarNode
            key={node.id}
            node={node}
            position={p}
            degree={degrees.get(node.id) ?? 1}
            emphasis={emphasis(node)}
            selected={selectedId === node.id}
            hovered={hoveredId === node.id}
            dragging={draggingId === node.id}
            onHover={(h) => onHover(h ? node.id : null)}
            onPointerDown={(e) => onNodePointerDown(node.id, e.point, e)}
          />
        );
      })}
      <OrbitControls
        ref={orbitRef}
        makeDefault
        enableDamping
        dampingFactor={0.05}
        minDistance={6.5}
        maxDistance={34}
        maxPolarAngle={Math.PI * 0.88}
        rotateSpeed={0.65}
      />
    </>
  );
}

export default function StarsPage() {
  const [tab, setTab] = useState<TabKey>('all');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);

  const [positions, setPositions] = useState(() =>
    clonePositionMap(runLayout(baguNodes, baguLinks)),
  );

  const resetLayout = useCallback(() => {
    setPositions(clonePositionMap(runLayout(baguNodes, baguLinks)));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedId(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const degrees = useMemo(() => nodeDegrees(baguLinks), []);
  const leaderboard = useMemo(() => {
    return [...baguNodes]
      .map((n) => ({ n, d: degrees.get(n.id) ?? 0 }))
      .sort((a, b) => b.d - a.d)
      .slice(0, 10);
  }, [degrees]);

  const selected = selectedId ? baguNodes.find((x) => x.id === selectedId) : null;
  const neighbors = useMemo(() => {
    if (!selectedId) return [];
    const set = new Set<string>();
    baguLinks.forEach((l) => {
      if (l.source === selectedId) set.add(l.target);
      if (l.target === selectedId) set.add(l.source);
    });
    return [...set].map((id) => baguNodes.find((x) => x.id === id)).filter(Boolean) as BaguNode[];
  }, [selectedId]);

  const tabs: { key: TabKey; label: string }[] = [
    { key: 'all', label: '全部' },
    { key: 'arch', label: BAGU_GROUP_LABEL.arch },
    { key: 'train', label: BAGU_GROUP_LABEL.train },
    { key: 'infer', label: BAGU_GROUP_LABEL.infer },
    { key: 'app', label: BAGU_GROUP_LABEL.app },
  ];

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'radial-gradient(ellipse 120% 80% at 50% 40%, #0c1929 0%, #020617 45%, #010409 100%)',
        color: '#e2e8f0',
        fontFamily: "'DM Sans', 'Plus Jakarta Sans', system-ui, sans-serif",
      }}
    >
      <Canvas
        camera={{ position: [0, 0.45, 18.5], fov: 52 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        style={{ width: '100%', height: '100%' }}
        onPointerMissed={() => setSelectedId(null)}
      >
        <Scene
          positions={positions}
          setPositions={setPositions}
          degrees={degrees}
          tab={tab}
          selectedId={selectedId}
          hoveredId={hoveredId}
          draggingId={draggingId}
          onHover={setHoveredId}
          onTapNode={(id) => setSelectedId((cur) => (cur === id ? null : id))}
          onDragState={setDraggingId}
        />
      </Canvas>

      {/* 顶栏 */}
      <div
        style={{
          position: 'absolute',
          top: 20,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 20,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 12,
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '10px 18px 10px 14px',
            borderRadius: 999,
            background: 'rgba(15,23,42,0.72)',
            border: '1px solid rgba(148,163,184,0.25)',
            backdropFilter: 'blur(14px)',
            boxShadow: '0 8px 32px rgba(0,0,0,0.45)',
          }}
        >
          <span style={{ fontSize: 20, marginRight: 4 }}>✦</span>
          <span style={{ fontWeight: 700, fontSize: 17, letterSpacing: '0.06em' }}>八股星图</span>
          <span style={{ opacity: 0.45, fontSize: 12, marginLeft: 4 }}>LLM</span>
        </div>
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            gap: 8,
            padding: '6px 8px',
            borderRadius: 999,
            background: 'rgba(15,23,42,0.55)',
            border: '1px solid rgba(148,163,184,0.2)',
            backdropFilter: 'blur(12px)',
          }}
        >
          {tabs.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              style={{
                border: 'none',
                cursor: 'pointer',
                padding: '8px 16px',
                borderRadius: 999,
                fontSize: 13,
                fontWeight: 600,
                color: tab === t.key ? '#0f172a' : '#94a3b8',
                background:
                  tab === t.key
                    ? 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)'
                    : 'transparent',
                boxShadow: tab === t.key ? '0 0 20px rgba(251,191,36,0.35)' : 'none',
                transition: 'background 0.2s, color 0.2s',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          top: 20,
          left: 20,
          zIndex: 20,
          display: 'flex',
          gap: 10,
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        <Link
          to="/"
          style={{
            color: '#94a3b8',
            textDecoration: 'none',
            fontSize: 13,
            padding: '8px 14px',
            borderRadius: 10,
            background: 'rgba(15,23,42,0.6)',
            border: '1px solid rgba(148,163,184,0.2)',
          }}
        >
          ← 返回
        </Link>
        <button
          type="button"
          onClick={resetLayout}
          style={{
            border: '1px solid rgba(148,163,184,0.25)',
            cursor: 'pointer',
            padding: '8px 14px',
            borderRadius: 10,
            fontSize: 13,
            fontWeight: 600,
            color: '#cbd5e1',
            background: 'rgba(15,23,42,0.6)',
          }}
        >
          重置布局
        </button>
        <span style={{ fontSize: 11, color: '#64748b' }}>Esc 关闭侧栏 · 轻点选中 · 拖移节点</span>
      </div>

      {/* 侧栏 */}
      {selected && (
        <div
          style={{
            position: 'absolute',
            top: 100,
            right: 20,
            bottom: 120,
            width: 320,
            maxHeight: 'calc(100vh - 140px)',
            zIndex: 20,
            borderRadius: 16,
            background: 'rgba(15,23,42,0.82)',
            border: '1px solid rgba(148,163,184,0.22)',
            backdropFilter: 'blur(16px)',
            boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <div style={{ padding: '18px 18px 12px', borderBottom: '1px solid rgba(148,163,184,0.15)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: BAGU_GROUP_COLOR[selected.group],
                  boxShadow: `0 0 12px ${BAGU_GROUP_COLOR[selected.group]}`,
                }}
              />
              <span style={{ fontSize: 20, fontWeight: 700 }}>{selected.label}</span>
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>
              {BAGU_GROUP_LABEL[selected.group]} · 关联 {neighbors.length} 个知识点
            </div>
          </div>
          <div style={{ padding: '12px 18px', flex: 1, overflowY: 'auto' }}>
            <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#64748b', marginBottom: 8 }}>
              面试追问
            </div>
            <ul style={{ margin: 0, paddingLeft: 18, color: '#cbd5e1', fontSize: 13, lineHeight: 1.55 }}>
              {selected.tips.map((tip, i) => (
                <li key={i} style={{ marginBottom: 8 }}>
                  {tip}
                </li>
              ))}
            </ul>
            {neighbors.length > 0 && (
              <>
                <div
                  style={{
                    fontSize: 11,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    color: '#64748b',
                    margin: '16px 0 8px',
                  }}
                >
                  相邻概念
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {neighbors.map((n) => (
                    <button
                      key={n.id}
                      type="button"
                      onClick={() => setSelectedId(n.id)}
                      style={{
                        border: `1px solid ${BAGU_GROUP_COLOR[n.group]}55`,
                        background: `${BAGU_GROUP_COLOR[n.group]}18`,
                        color: '#e2e8f0',
                        borderRadius: 999,
                        padding: '6px 12px',
                        fontSize: 12,
                        cursor: 'pointer',
                      }}
                    >
                      {n.label}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* 底栏图例 */}
      <div
        style={{
          position: 'absolute',
          left: 20,
          right: 20,
          bottom: 16,
          zIndex: 20,
          padding: '14px 18px',
          borderRadius: 14,
          background: 'rgba(15,23,42,0.75)',
          border: '1px solid rgba(148,163,184,0.2)',
          backdropFilter: 'blur(14px)',
        }}
      >
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 10, letterSpacing: '0.06em' }}>
          关联最多（度数）
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
            gap: '8px 16px',
          }}
        >
          {leaderboard.map(({ n, d }) => (
            <button
              key={n.id}
              type="button"
              onClick={() => setSelectedId(n.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                border: 'none',
                borderRadius: 8,
                cursor: 'pointer',
                padding: '6px 8px',
                margin: '-2px -8px',
                textAlign: 'left',
                background:
                  selectedId === n.id
                    ? 'linear-gradient(90deg, rgba(251,191,36,0.35) 0%, rgba(245,158,11,0.08) 100%)'
                    : 'transparent',
                boxShadow:
                  selectedId === n.id ? 'inset 0 0 0 1px rgba(251,191,36,0.45)' : 'none',
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  flexShrink: 0,
                  background: BAGU_GROUP_COLOR[n.group],
                  boxShadow: `0 0 8px ${BAGU_GROUP_COLOR[n.group]}88`,
                }}
              />
              <span style={{ fontSize: 13, color: '#e2e8f0', flex: 1 }}>{n.label}</span>
              <span style={{ fontSize: 12, color: '#64748b', fontVariantNumeric: 'tabular-nums' }}>{d}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
