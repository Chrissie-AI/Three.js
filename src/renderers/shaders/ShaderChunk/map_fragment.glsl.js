export default /* glsl */`
#ifdef USE_MAP

vec4 texelColor = texture2D( map, vec2( vUv.x / vUv.z, vUv.y )  );

	texelColor = mapTexelToLinear( texelColor );
	diffuseColor *= texelColor;

#endif
`;
