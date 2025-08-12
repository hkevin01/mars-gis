// Copy Cesium static assets (Workers, ThirdParty, Assets, Widgets) to public/cesium
// so they are served by CRA/Nginx at runtime. Requires Node 16+ for fs.cp.

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');

async function copyDir(src, dest) {
  try {
    await fsp.mkdir(dest, { recursive: true });
    if (fs.cp) {
      // Node 16+
      await fsp.cp(src, dest, { recursive: true });
      return;
    }
  } catch (_) {}

  // Fallback manual copy
  const entries = await fsp.readdir(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      await copyDir(srcPath, destPath);
    } else if (entry.isSymbolicLink()) {
      const link = await fsp.readlink(srcPath);
      await fsp.symlink(link, destPath);
    } else {
      await fsp.copyFile(srcPath, destPath);
    }
  }
}

async function main() {
  try {
    const cesiumBuild = path.join(
      __dirname,
      '..',
      'node_modules',
      'cesium',
      'Build',
      'Cesium'
    );
    const publicCesium = path.join(__dirname, '..', 'public', 'cesium');
    const dirs = ['Assets', 'Widgets', 'Workers', 'ThirdParty'];

    for (const d of dirs) {
      const src = path.join(cesiumBuild, d);
      const dest = path.join(publicCesium, d);
      if (fs.existsSync(src)) {
        // eslint-disable-next-line no-console
        console.log(`Copying Cesium ${d} to ${dest}`);
        await copyDir(src, dest);
      }
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('Cesium asset copy skipped or failed:', err && err.message);
  }
}

main();
