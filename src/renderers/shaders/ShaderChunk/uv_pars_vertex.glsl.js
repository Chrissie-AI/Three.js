export default /* glsl */`
#ifdef USE_UV

	#ifdef UVS_VERTEX_ONLY

		vec2 vUv;

	#else

		varying vec3 vUv;

	#endif

	uniform mat3 uvTransform;

#endif
`;

