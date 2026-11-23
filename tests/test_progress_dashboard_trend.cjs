"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../tools/progress_dashboard.html"), "utf8");
const trendSource = html.slice(html.indexOf("function trend() {"), html.indexOf("function exportCsv() {"));
const original = { v: 0.5, n: 1, d: 2, f: "original" };
const changed = { v: 0.75, n: 3, d: 4, f: "changed" };
const unavailable = { v: null, f: "original" };

class Element {
  constructor(tag) {
    this.tag = tag;
    this.attributes = {};
    this.children = [];
  }

  setAttribute(name, value) { this.attributes[name] = value; }
  append(child) { this.children.push(child); }
  prepend(child) { this.children.unshift(child); }
  replaceChildren() { this.children = []; }
}

function renderTrend(samples) {
  const svg = new Element("svg");
  vm.runInNewContext(`${trendSource}\ntrend();`, {
    HIST: samples.map(sample => ({ metrics: sample === undefined ? {} : { "gpu.current": sample } })),
    document: { getElementById: () => svg, createElementNS: (_namespace, tag) => new Element(tag) },
    css: name => name,
    pct: String,
  });
  const line = svg.children.find(child => child.tag === "path");
  const points = svg.children.filter(child => child.tag === "circle");
  return {
    commands: line.attributes.d.match(/[ML]/g).join(""),
    hollow: points.map(point => point.attributes.fill === "panel"),
    changeLabels: points.map(point => point.children[0].textContent.includes("formula changed here")),
  };
}

test("draws connected values within a formula and marks adjacent formula changes", () => {
  const chart = renderTrend([original, original, changed, changed]);
  assert.equal(chart.commands, "MLML");
  assert.deepEqual(chart.hollow, [false, false, true, false]);
  assert.deepEqual(chart.changeLabels, chart.hollow);
});

test("breaks the line across missing and unavailable values with the same formula", () => {
  const chart = renderTrend([original, undefined, original, unavailable, original]);
  assert.equal(chart.commands, "MMM");
  assert.deepEqual(chart.hollow, [false, false, false]);
  assert.deepEqual(chart.changeLabels, chart.hollow);
});

test("marks formula changes across gaps and resumes the new line segment", () => {
  for (const gap of [undefined, unavailable]) {
    const chart = renderTrend([original, gap, changed, changed]);
    assert.equal(chart.commands, "MML");
    assert.deepEqual(chart.hollow, [false, true, false]);
    assert.deepEqual(chart.changeLabels, chart.hollow);
  }
});
