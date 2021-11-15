export default /*  glsl */`
#ifdef USE_UV

	vUv = ( uvTransform * vec3( uv ) );

#endif
`;
