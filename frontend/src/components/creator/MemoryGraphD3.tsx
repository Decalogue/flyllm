import React, { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import type { GraphNode, GraphLink, MemoryGraphData } from './memoryGraphData';
import { CREATOR_THEME } from './creatorTheme';

const T = CREATOR_THEME;

export interface MemoryGraphD3Props {
  data: MemoryGraphData;
  width: number;
  height: number;
  onNodeClick?: (node: GraphNode) => void;
  className?: string;
}

type SimNode = GraphNode & d3.SimulationNodeDatum;
type SimLink = d3.SimulationLinkDatum<SimNode> & { source: string; target: string; relation?: string };

const COLORS: Record<string, string> = {
  entity: T.accent,
  fact: T.brandLight,
  atom: '#f59e0b',
};

export const MemoryGraphD3: React.FC<MemoryGraphD3Props> = ({
  data,
  width,
  height,
  onNodeClick,
  className,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  const run = useCallback(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;
    const nodes: SimNode[] = data.nodes.map((n, i) => {
      const angle = (i / Math.max(1, data.nodes.length)) * Math.PI * 2;
      return {
        ...n,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      };
    });
    const links: SimLink[] = data.links.map((l) => ({ ...l }));

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (ev) => g.attr('transform', ev.transform));
    svg.call(zoom as any);

    const link = g
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', 'rgba(255, 75, 47, 0.25)')
      .attr('stroke-width', 1.5);

    const node = g
      .append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(
        d3
          .drag<SVGGElement, SimNode>()
          .on('start', (ev) => {
            if (!ev.subject.fx) ev.subject.fx = ev.subject.x;
            if (!ev.subject.fy) ev.subject.fy = ev.subject.y;
          })
          .on('drag', (ev) => {
            ev.subject.fx = ev.x;
            ev.subject.fy = ev.y;
          })
          .on('end', (ev) => {
            ev.subject.fx = null;
            ev.subject.fy = null;
          }) as any
      )
      .on('click', (_, d) => onNodeClick?.(d as GraphNode));

    node.append('circle').attr('r', 10).attr('fill', (d) => COLORS[d.type] || T.textDim);

    node
      .append('text')
      .attr('dx', 14)
      .attr('dy', 4)
      .attr('fill', T.text)
      .attr('font-size', 10)
      .text((d) => d.label);

    const sim = d3
      .forceSimulation<SimNode>(nodes)
      .force('link', d3.forceLink<SimNode, SimLink>(links).id((d: any) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2));

    sim.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    simRef.current = sim;
  }, [data, width, height, onNodeClick]);

  useEffect(() => {
    run();
    return () => {
      simRef.current?.stop();
      simRef.current = null;
    };
  }, [run]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className={className}
      style={{ display: 'block', background: T.bgGraph, borderRadius: 8 }}
    />
  );
};
