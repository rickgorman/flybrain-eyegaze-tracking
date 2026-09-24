import { geomMatchesStored } from './gaze.mjs';

/** True while an in-flight camera init still owns the current generation counter. */
export function cameraSessionOwned(sessionGen, currentGen) {
  return sessionGen === currentGen;
}

/**
 * Whether accuracy protocol may run or persist results (model required by caller context).
 */
export function accuracyProtocolAllowed({ model, geomStale, storedGeom, currentGeom }) {
  if (!model) return false;
  if (geomStale) return false;
  if (!storedGeom) return false;
  return geomMatchesStored(storedGeom, currentGeom);
}

