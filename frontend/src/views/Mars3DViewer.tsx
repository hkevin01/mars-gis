import {
    Cartesian3,
    Credit,
    Ellipsoid,
    UrlTemplateImageryProvider,
    Viewer,
    WebMercatorTilingScheme,
} from 'cesium';
import React, { useEffect, useRef } from 'react';

const Mars3DViewer: React.FC = () => {
	const containerRef = useRef<HTMLDivElement | null>(null);
	const viewerRef = useRef<Viewer | null>(null);

	useEffect(() => {
		if (!containerRef.current) return;

		// Mars radii in meters (IAU 2000): Equatorial 3396.19 km, Polar 3376.2 km
		const marsEllipsoid = new Ellipsoid(3396190.0, 3396190.0, 3376200.0);

		// Build backend tile URL from env if available; otherwise leave undefined to force fallback immediately.
		const apiBase = (process.env.REACT_APP_API_URL || '').replace(/\/$/, '');
		const apiTilesUrl = apiBase ? `${apiBase}/tiles/{z}/{x}/{y}.png` : '';

		// Start with fallback by default; we'll replace only if backend tile check succeeds.
		let imageryProvider: UrlTemplateImageryProvider = new UrlTemplateImageryProvider({
			url: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg',
			tilingScheme: new WebMercatorTilingScheme({ ellipsoid: marsEllipsoid }),
			credit: new Credit('NASA Mars Trek (fallback)')
		});

		if (apiTilesUrl) {
			(async () => {
				try {
					const controller = new AbortController();
					const timeout = setTimeout(() => controller.abort(), 2500);
					const testUrl = apiTilesUrl.replace('{z}', '0').replace('{x}', '0').replace('{y}', '0');
					const res = await fetch(testUrl, { method: 'HEAD', signal: controller.signal });
					clearTimeout(timeout);
					if (!res.ok) throw new Error('tile head not ok');
					// Success: switch to backend imagery
					imageryProvider = new UrlTemplateImageryProvider({
						url: apiTilesUrl,
						tilingScheme: new WebMercatorTilingScheme({ ellipsoid: marsEllipsoid }),
						credit: new Credit('MOLA/OPM (backend)')
					});
					if (viewerRef.current) {
						viewerRef.current.imageryLayers.removeAll();
						viewerRef.current.imageryLayers.addImageryProvider(imageryProvider);
					}
				} catch (_) {
					// keep fallback
				}
			})();
		}

			const viewer = new Viewer(containerRef.current, {
			baseLayerPicker: false,
			geocoder: false,
			homeButton: false,
			navigationHelpButton: false,
			timeline: false,
			animation: false,
			fullscreenButton: false,
			sceneModePicker: false,
			skyAtmosphere: false,
			shouldAnimate: false,
		});

		// NOTE: Cesium Globe.ellipsoid has only a getter in recent Cesium versions; we avoid direct reassignment.
		// Mars-specific ellipsoid is applied through the tiling scheme & cartographic conversions where needed.

			// Use selected imagery provider (fallback initially)
			try {
				viewer.imageryLayers.removeAll();
				viewer.imageryLayers.addImageryProvider(imageryProvider);
			} catch (_) { /* ignore */ }

		// Aim camera towards Mars center with a decent height
		viewer.camera.setView({
			destination: Cartesian3.fromDegrees(0, 0, 5_000_000),
		});

		viewerRef.current = viewer;
		return () => {
			viewerRef.current?.destroy();
			viewerRef.current = null;
		};
	}, []);

	return (
		<div className="w-full h-full">
			<div ref={containerRef} className="w-full h-full" />
		</div>
	);
};

export default Mars3DViewer;
