import {
    Cartesian3,
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
		const marsEllipsoid = new Ellipsoid(
			3396190.0,
			3396190.0,
			3376200.0
		);

			const imageryProvider = new UrlTemplateImageryProvider({
			url: '/api/v1/tiles/{z}/{x}/{y}.png',
			tilingScheme: new WebMercatorTilingScheme({ ellipsoid: marsEllipsoid }),
			credit: 'MOLA/OPM via proxy',
		});

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

		// Replace global ellipsoid with Mars for camera/cartographic conversions
		viewer.scene.globe.ellipsoid = marsEllipsoid;

				// Use our proxy imagery provider as the only base layer
				try {
					viewer.imageryLayers.removeAll();
					viewer.imageryLayers.addImageryProvider(imageryProvider);
				} catch (e) {
					// noop - viewer may still initialize with default base layer
				}

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
