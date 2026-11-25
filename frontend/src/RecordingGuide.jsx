// Recording steps shown on the capture page. For the RGB-only mesh demo (person turning in place).
export default function RecordingGuide() {
  return (
    <details className="card wide guide">
      <summary>How to record a better scan (read before filming)</summary>
      <div className="cols">
        <div>
          <h3>Camera</h3>
          <ul>
            <li>Fix the iPad on a rigid stand. Do not hold it.</li>
            <li>Keep the whole body in the picture with clear space above the head, below the feet and at both sides.</li>
            <li>Lock focus and exposure (AE/AF lock), turn HDR video off, do not zoom.</li>
          </ul>
          <h3>Room</h3>
          <ul>
            <li>Plain wall, even light, no strong shadows, nobody else in the picture.</li>
            <li>Record a few seconds of the empty background first, in a separate clip.</li>
          </ul>
        </div>
        <div>
          <h3>Person</h3>
          <ul>
            <li>Tight, plain clothes that contrast with the wall.</li>
            <li>Arms held away from the body, legs apart. Keep the same pose the whole time.</li>
            <li>Mark the foot position on the floor.</li>
            <li>Turn one full circle in small equal steps; stand completely still at each stop.</li>
          </ul>
          <h3>For checking accuracy</h3>
          <ul>
            <li>Measure height, chest, waist and hip with a tape and write them down.</li>
          </ul>
        </div>
      </div>
      <p className="muted">These steps reduce errors. They do not make the experimental mesh accurate or give it a real size.</p>
    </details>
  );
}
