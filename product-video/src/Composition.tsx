import { Composition } from "remotion";
import { CinematicVideo } from "./CinematicVideo";

export const MyComposition: React.FC = () => (
  <Composition
    id="AutoMLProductVideo"
    component={CinematicVideo}
    durationInFrames={720}
    fps={30}
    width={1280}
    height={720}
  />
);
