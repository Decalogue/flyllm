import React, { useRef, useMemo, useState, useCallback } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { Html, Line, OrbitControls } from '@react-three/drei';
import type { GraphNode, GraphLink, MemoryGraphData } from './memoryGraphData';
import { CREATOR_THEME } from './creatorTheme';

const T = CREATOR_THEME;

const COLORS: Record<string, string> = {
  entity: '#ff4b2f',
  fact: '#ffa7a7',
  atom: '#f59e0b',
};

const TYPE_ICONS: Record<string, string> = {
  entity: '◆',
  fact: '◇',
  atom: '◎',
};

const TYPE_LABELS: Record<string, string> = {
  entity: '实体',
  fact: '事实',
  atom: '原子笔记',
};

/** 按类型返回节点几何体，使实体/事实/原子笔记在图中形状区分明显 */
function getNodeGeometry(type: string, baseSize: number): THREE.BufferGeometry {
  switch (type) {
    case 'entity':
      return new THREE.OctahedronGeometry(baseSize, 0);
    case 'fact':
      return new THREE.CylinderGeometry(baseSize * 0.85, baseSize * 0.85, baseSize * 0.5, 24);
    default:
      return new THREE.SphereGeometry(baseSize, 28, 28);
  }
}

export interface MemoryGraphThreeProps {
  data: MemoryGraphData;
  width: number;
  height: number;
  onNodeClick?: (node: GraphNode) => void;
  className?: string;
  /** 首页 intro 用，节点与字体更大 */
  variant?: 'intro' | 'default';
}

function useLayout(nodes: GraphNode[], links: GraphLink[]) {
  return useMemo(() => {
    const pos = new Map<string, THREE.Vector3>();
    const n = nodes.length;
    const radius = 3.5;
    nodes.forEach((node, i) => {
      const angle = (i / Math.max(1, n)) * Math.PI * 2;
      const r = radius + (i % 3) * 0.4;
      const h = ((i % 5) / 5 - 0.5) * 2.5;
      pos.set(node.id, new THREE.Vector3(
        Math.cos(angle) * r,
        h,
        Math.sin(angle) * r
      ));
    });
    return pos;
  }, [nodes, links]);
}

function BackgroundParticles() {
  const ref = useRef<THREE.Points>(null);
  const count = 150;
  const [positions] = useState(() => {
    const a = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      a[i * 3] = (Math.random() - 0.5) * 24;
      a[i * 3 + 1] = (Math.random() - 0.5) * 24;
      a[i * 3 + 2] = (Math.random() - 0.5) * 24;
    }
    return a;
  });

  useFrame((state) => {
    if (ref.current) ref.current.rotation.y = state.clock.elapsedTime * 0.02;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        color={T.accent}
        transparent
        opacity={0.35}
        sizeAttenuation
      />
    </points>
  );
}

function ConnectionLine({
  start,
  end,
  isActive,
  color,
}: {
  start: THREE.Vector3;
  end: THREE.Vector3;
  isActive: boolean;
  color: string;
}) {
  const points = useMemo(() => [start, end] as [THREE.Vector3, THREE.Vector3], [start, end]);
  return (
    <Line
      points={points}
      color={color}
      lineWidth={isActive ? 2.5 : 1}
      transparent
      opacity={isActive ? 0.85 : 0.35}
    />
  );
}

function NodeMesh({
  node,
  position,
  isActive,
  isHovered,
  onClick,
  onHover,
  variant = 'default',
}: {
  node: GraphNode;
  position: THREE.Vector3;
  isActive: boolean;
  isHovered: boolean;
  onClick: () => void;
  onHover: (h: boolean) => void;
  variant?: 'intro' | 'default';
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const color = COLORS[node.type] ?? T.accent;
  const baseSize = variant === 'intro' ? 0.38 : 0.28;
  const labelFontSize = variant === 'intro' ? 14 : 11;
  const labelPadding = variant === 'intro' ? '6px 12px' : '4px 10px';
  const labelDistanceFactor = variant === 'intro' ? 16 : 12;
  const nodeGeometry = useMemo(
    () => getNodeGeometry(node.type, baseSize),
    [node.type, baseSize]
  );

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (meshRef.current) {
      meshRef.current.position.y = position.y + Math.sin(t * 1.8 + position.x) * 0.08;
      if (isActive) meshRef.current.rotation.y += 0.015;
    }
    if (glowRef.current) {
      const s = isHovered || isActive ? 1.4 + Math.sin(t * 3) * 0.15 : 1.15;
      glowRef.current.scale.setScalar(s);
    }
    if (ringRef.current && isActive) {
      ringRef.current.rotation.x += 0.01;
      ringRef.current.rotation.y += 0.018;
    }
  });

  return (
    <group position={position}>
      <mesh ref={glowRef} scale={1.15}>
        <sphereGeometry args={[baseSize * 1.6, 16, 16]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={isActive ? 0.35 : 0.12}
          side={THREE.BackSide}
        />
      </mesh>
      {isActive && (
        <mesh ref={ringRef}>
          <torusGeometry args={[baseSize * 1.8, 0.02, 8, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.6} />
        </mesh>
      )}
      <mesh
        ref={meshRef}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
        onPointerOver={(e) => { e.stopPropagation(); onHover(true); }}
        onPointerOut={() => onHover(false)}
        scale={isHovered ? 1.15 : 1}
      >
        <primitive object={nodeGeometry} attach="geometry" />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActive ? 0.5 : 0.2}
          roughness={0.3}
          metalness={0.65}
        />
      </mesh>
      <Html distanceFactor={labelDistanceFactor} position={[0, baseSize + 0.35, 0]} center>
        <div
          style={{
            padding: labelPadding,
            borderRadius: variant === 'intro' ? 12 : 999,
            fontSize: labelFontSize,
            fontWeight: T.fontWeightSemibold,
            whiteSpace: variant === 'intro' && (node.brief || node.source) ? 'normal' : 'nowrap',
            cursor: 'pointer',
            background: isActive ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.7)',
            color: isActive ? '#0a0a0a' : '#fafafa',
            border: `2px solid ${color}`,
            transform: isActive ? 'scale(1.08)' : undefined,
            transition: 'background 0.2s, color 0.2s, transform 0.2s',
            textAlign: 'center',
            minWidth: variant === 'intro' ? 72 : undefined,
            maxWidth: variant === 'intro' ? 120 : undefined,
          }}
        >
          <div>{TYPE_ICONS[node.type]} {node.label}</div>
          {variant === 'intro' && (node.brief || node.source) && (
            <div style={{ fontSize: labelFontSize - 2, fontWeight: 400, opacity: 0.85, marginTop: 2 }}>
              {node.brief ?? node.source}
            </div>
          )}
        </div>
      </Html>
    </group>
  );
}

function Scene({
  nodes,
  links,
  positions,
  activeId,
  hoveredId,
  onNodeClick,
  onHover,
  variant = 'default',
}: {
  nodes: GraphNode[];
  links: GraphLink[];
  positions: Map<string, THREE.Vector3>;
  activeId: string | null;
  hoveredId: string | null;
  onNodeClick: (n: GraphNode) => void;
  onHover: (id: string | null) => void;
  variant?: 'intro' | 'default';
}) {
  const isLinkActive = useCallback(
    (sid: string, tid: string) => {
      if (activeId === sid || activeId === tid) return true;
      if (hoveredId && (hoveredId === sid || hoveredId === tid)) return true;
      return false;
    },
    [activeId, hoveredId]
  );

  return (
    <>
      <ambientLight intensity={0.45} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#ffffff" />
      <pointLight position={[-8, -8, 6]} intensity={0.5} color={T.accent} />
      <BackgroundParticles />
      {links.map((l) => {
        const sid = typeof l.source === 'string' ? l.source : (l.source as { id?: string }).id;
        const tid = typeof l.target === 'string' ? l.target : (l.target as { id?: string }).id;
        if (!sid || !tid) return null;
        const start = positions.get(sid);
        const end = positions.get(tid);
        if (!start || !end) return null;
        const color = COLORS[nodes.find((n) => n.id === sid)?.type ?? 'entity'] ?? T.accent;
        return (
          <ConnectionLine
            key={`${sid}-${tid}`}
            start={start}
            end={end}
            isActive={isLinkActive(sid, tid)}
            color={color}
          />
        );
      })}
      {nodes.map((node) => {
        const pos = positions.get(node.id);
        if (!pos) return null;
        return (
          <NodeMesh
            key={node.id}
            node={node}
            position={pos}
            isActive={activeId === node.id}
            isHovered={hoveredId === node.id}
            onClick={() => onNodeClick(node)}
            onHover={(h) => onHover(h ? node.id : null)}
            variant={variant}
          />
        );
      })}
      <OrbitControls
        enableDamping
        dampingFactor={0.06}
        minDistance={4}
        maxDistance={20}
        maxPolarAngle={Math.PI * 0.85}
      />
    </>
  );
}

export const MemoryGraphThree: React.FC<MemoryGraphThreeProps> = ({
  data,
  width,
  height,
  onNodeClick,
  className,
  variant = 'default',
}) => {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const positions = useLayout(data.nodes, data.links);
  const isIntro = variant === 'intro';

  const handleClick = useCallback(
    (node: GraphNode) => {
      setActiveId(node.id);
      onNodeClick?.(node);
    },
    [onNodeClick]
  );

  if (!data.nodes.length) {
    return (
      <div
        className={className}
        style={{
          width,
          height,
          borderRadius: 8,
          overflow: 'hidden',
          background: T.bgGraph,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: T.textDim,
          fontSize: 13,
        }}
      >
        暂无图谱数据
      </div>
    );
  }

  return (
    <div
      className={className}
      style={{
        width,
        height,
        borderRadius: 8,
        overflow: 'hidden',
        background: `linear-gradient(180deg, ${T.bgGraphSolid} 0%, #1a1a2e 100%)`,
        position: 'relative',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: isIntro ? 12 : 10,
          left: isIntro ? 12 : 10,
          zIndex: 10,
          pointerEvents: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: isIntro ? 12 : 10,
          padding: isIntro ? '10px 16px' : '8px 14px',
          background: 'rgba(0,0,0,0.5)',
          backdropFilter: 'blur(12px)',
          borderRadius: 12,
          border: `1px solid ${T.accentBorder}`,
        }}
      >
        <span style={{ width: isIntro ? 10 : 8, height: isIntro ? 10 : 8, borderRadius: '50%', background: '#22c55e' }} />
        <span style={{ fontSize: isIntro ? 15 : 13, fontWeight: T.fontWeightSemibold, color: T.accent, fontFamily: 'monospace' }}>
          云记忆
        </span>
      </div>
      <div
        style={{
          position: 'absolute',
          top: isIntro ? 12 : 10,
          right: isIntro ? 12 : 10,
          zIndex: 10,
          pointerEvents: 'none',
          padding: isIntro ? '10px 16px' : '8px 14px',
          background: 'rgba(0,0,0,0.5)',
          backdropFilter: 'blur(12px)',
          borderRadius: 12,
          border: `1px solid ${T.border}`,
        }}
      >
        <span style={{ fontSize: isIntro ? 15 : 13, color: T.textMuted, fontFamily: 'monospace' }}>
          节点 {data.nodes.length}
        </span>
      </div>
      <Canvas
        camera={{ position: [8, 4, 10], fov: 55 }}
        gl={{ antialias: true, alpha: true }}
        style={{ width, height }}
      >
        <Scene
          nodes={data.nodes}
          links={data.links}
          positions={positions}
          activeId={activeId}
          hoveredId={hoveredId}
          onNodeClick={handleClick}
          onHover={setHoveredId}
          variant={variant}
        />
      </Canvas>
      <div
        style={{
          position: 'absolute',
          bottom: isIntro ? 12 : 10,
          left: isIntro ? 12 : 10,
          right: isIntro ? 12 : 10,
          zIndex: 10,
          pointerEvents: 'none',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: isIntro ? 16 : 14,
          padding: isIntro ? '12px 16px' : '10px 14px',
          background: 'rgba(0,0,0,0.5)',
          backdropFilter: 'blur(12px)',
          borderRadius: 12,
          border: `1px solid ${T.border}`,
        }}
      >
        {(['entity', 'fact', 'atom'] as const).map((type) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: isIntro ? 10 : 8 }}>
            <span style={{ fontSize: isIntro ? 18 : 16, color: COLORS[type] }}>{TYPE_ICONS[type]}</span>
            <span style={{ fontSize: isIntro ? 14 : 13, color: T.textMuted }}>{TYPE_LABELS[type]}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
