// Match apitofsim.relayout_labels: rotate -45 degrees, visit peaks from
// right to left, then move each box up in five-pixel diagonal steps until
// it clears the placed labels and the spectrum. Recompute in screen space.
cancelAnimationFrame(source._labelFrame);
source._labelFrame = requestAnimationFrame(() => {
  const width = plot.inner_width,
    height = plot.inner_height;
  const xmin = plot.x_range.start,
    xmax = plot.x_range.end;
  const ymin = plot.y_range.start,
    ymax = plot.y_range.end;
  if (!(width > 0 && height > 0 && xmax > xmin && ymax > ymin)) return;
  const data = source.data;
  const context = document.createElement("canvas").getContext("2d");
  context.font = "10pt sans-serif";
  const c = Math.SQRT1_2;
  const rotate = (x, y) => [(x + y) * c, (y - x) * c];
  const unrotate = (x, y) => [(x - y) * c, (x + y) * c];
  const sx = (x) => ((x - xmin) * width) / (xmax - xmin);
  const sy = (y) => ((y - ymin) * height) / (ymax - ymin);
  const dx = (x) => xmin + (x * (xmax - xmin)) / width;
  const dy = (y) => ymin + (y * (ymax - ymin)) / height;
  const visible = data.mass
    .map((_, i) => i)
    .filter(
      (i) =>
        data.mass[i] >= xmin &&
        data.mass[i] <= xmax &&
        data.intensity[i] >= ymin &&
        data.intensity[i] <= ymax,
    )
    .sort((a, b) => data.mass[b] - data.mass[a] || a - b);
  const spikes = data.mass.map((mass, i) => [
    rotate(sx(mass), sy(0)),
    rotate(sx(mass), sy(data.intensity[i])),
  ]);
  const boxes = [];
  const xs = [],
    ys = [];
  const x = Array(data.mass.length).fill(NaN);
  const y = Array(data.mass.length).fill(NaN);
  // Segment/rectangle intersection in the rotated coordinate system.
  function intersects(box, line) {
    let lo = 0,
      hi = 1;
    for (let axis = 0; axis < 2; axis++) {
      const start = line[0][axis],
        delta = line[1][axis] - start;
      if (Math.abs(delta) < 1e-10) {
        if (start < box[axis] || start > box[axis + 2]) return false;
      } else {
        const a = (box[axis] - start) / delta;
        const b = (box[axis + 2] - start) / delta;
        lo = Math.max(lo, Math.min(a, b));
        hi = Math.min(hi, Math.max(a, b));
        if (lo > hi) return false;
      }
    }
    return true;
  }
  let top = height,
    right = width;
  for (const i of visible) {
    const px = sx(data.mass[i]),
      py = sy(data.intensity[i]);
    const [rx, ry] = rotate(px, py);
    const metrics = context.measureText(data.text[i]);
    // Padding covers glyph bearings, descenders, and a gap between labels.
    const w = metrics.width + 4;
    const h =
      metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent + 4;
    let offset = 5,
      box;
    while (true) {
      box = [
        rx + offset - 2,
        ry + offset - 2,
        rx + offset + w,
        ry + offset + h,
      ];
      const overlaps = boxes.some(
        (b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1],
      );
      if (!overlaps && !spikes.some((line) => intersects(box, line))) break;
      offset += 5;
    }
    boxes.push(box);
    const [lx, ly] = unrotate(rx + offset, ry + offset);
    x[i] = dx(lx);
    y[i] = dy(ly);
    xs.push([data.mass[i], data.mass[i], dx(lx + 5)]);
    ys.push([data.intensity[i], y[i], dy(ly + 5)]);
    for (const [bx, by] of [
      [box[0], box[1]],
      [box[0], box[3]],
      [box[2], box[1]],
      [box[2], box[3]],
    ]) {
      const [cx, cy] = unrotate(bx, by);
      top = Math.max(top, cy);
      right = Math.max(right, cx);
    }
  }
  // Keep the data frame the same size while adding room for displaced labels.
  const extraTop = Math.max(0, Math.ceil(top - height) + 10);
  const extraRight = Math.max(0, Math.ceil(right - width) + 10);
  if (extraTop > plot.min_border_top) {
    plot.height += extraTop - plot.min_border_top;
    plot.min_border_top = extraTop;
  }
  if (extraRight > plot.min_border_right) {
    plot.width += extraRight - plot.min_border_right;
    plot.min_border_right = extraRight;
  }
  source.data = { ...data, x, y };
  leaders.data = { xs, ys };
});
