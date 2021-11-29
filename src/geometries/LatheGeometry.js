import { Float32BufferAttribute } from '../core/BufferAttribute.js';
import { BufferGeometry } from '../core/BufferGeometry.js';
import { Vector3 } from '../math/Vector3.js';
import { Vector2 } from '../math/Vector2.js';
import * as MathUtils from '../math/MathUtils.js';

class LatheGeometry extends BufferGeometry {

	constructor( points = [ new Vector2( 0, 0.5 ), new Vector2( 0.5, 0 ), new Vector2( 0, - 0.5 ) ], segments = 12, phiStart = 0, phiLength = Math.PI * 2 ) {

		super();

		this.type = 'LatheGeometry';

		this.parameters = {
			points: points,
			segments: segments,
			phiStart: phiStart,
			phiLength: phiLength
		};

		segments = Math.floor( segments );

		// clamp phiLength so it's in range of [ 0, 2PI ]

		phiLength = MathUtils.clamp( phiLength, 0, Math.PI * 2 );

		// buffers

		const indices = [];
		const vertices = [];
		const initNormals = [];
		const normals = [];
		const uvs = [];

		// helper variables

		const inverseSegments = 1.0 / segments;
      const normal = new Vector3();
		var prevNormal = new Vector3();
		var curNormal = new Vector3();
		const vertex = new Vector3();
		const uv = new Vector3();
		var scale = 1.0;
		var dx = 0;
		var dy = 0;

		// pre-compute normals for initial "meridian"

		for ( let j = 0; j <= ( points.length - 1 ); j ++ ) {

			switch ( j ) {

				case 0:		// special handling for 1st vertex on path

					dx = points[ j + 1 ].x - points[ j ].x;
					dy = points[ j + 1 ].y - points[ j ].y;

					normal.x = dy * 1.0;
					normal.y = -dx;
					normal.z = dy * 0.0;

					prevNormal.copy( normal );

					normal.normalize();

					initNormals.push( normal.x, normal.y, normal.z );

					break;

				case ( points.length - 1 ):      // special handling for last Vertex on path

					initNormals.push( prevNormal.x, prevNormal.y, prevNormal.z );

					break;

				default:      // default handling for all vertices in between

					dx = points[ j + 1 ].x - points[ j ].x;
					dy = points[ j + 1 ].y - points[ j ].y;

					normal.x = dy * 1.0;
					normal.y = -dx;
					normal.z = dy * 0.0;

					curNormal.copy( normal );

					normal.x += prevNormal.x;
					normal.y += prevNormal.y;
					normal.z += prevNormal.z;

					normal.normalize();

					initNormals.push( normal.x, normal.y, normal.z );

					prevNormal.copy( curNormal );

			}

		}

		// generate vertices, uvs and normals

		for ( let i = 0; i <= segments; i ++ ) {

			const phi = phiStart + i * inverseSegments * phiLength;

			const sin = Math.sin( phi );
			const cos = Math.cos( phi );

			for ( let j = 0; j <= ( points.length - 1 ); j ++ ) {

				scale = points[ j ].x / points[ 0 ].x;

				// vertex

				vertex.x = points[ j ].x * sin;
				vertex.y = points[ j ].y;
				vertex.z = points[ j ].x * cos;

				vertices.push( vertex.x, vertex.y, vertex.z );

				// uv

				uv.x = ( i / segments ) * scale;
				uv.y = j / ( points.length - 1 );
				uv.z = scale;

				uvs.push( uv.x, uv.y, uv.z );

				// normal

				let x = initNormals[ 3 * j + 0 ] * sin;
				let y = initNormals[ 3 * j + 1 ];
				let z = initNormals[ 3 * j + 0 ] * cos;

				normals.push( x, y, z );

			}

		}

		// indices

		for ( let i = 0; i < segments; i ++ ) {

			for ( let j = 0; j < ( points.length - 1 ); j ++ ) {

				const base = j + i * points.length;

				const a = base;
				const b = base + points.length;
				const c = base + points.length + 1;
				const d = base + 1;

				// faces

				indices.push( a, b, d );
				indices.push( b, c, d );

			}

		}

		// build geometry

		this.setIndex( indices );
		this.setAttribute( 'position', new Float32BufferAttribute( vertices, 3 ) );
		this.setAttribute( 'uv', new Float32BufferAttribute( uvs, 3 ) );
		this.setAttribute( 'normal', new Float32BufferAttribute( normals, 3 ) );

		// legacy normals generation and seam handling now obsolete
/*
		// generate normals

		this.computeVertexNormals();

		// if the geometry is closed, we need to average the normals along the seam.
		// because the corresponding vertices are identical (but still have different UVs).

		if ( phiLength === Math.PI * 2 ) {

			const normals = this.attributes.normal.array;
			const n1 = new Vector3();
			const n2 = new Vector3();
			const n = new Vector3();

			// this is the buffer offset for the last line of vertices

			const base = segments * points.length * 3;

			for ( let i = 0, j = 0; i < points.length; i ++, j += 3 ) {

				// select the normal of the vertex in the first line

				n1.x = normals[ j + 0 ];
				n1.y = normals[ j + 1 ];
				n1.z = normals[ j + 2 ];

				// select the normal of the vertex in the last line

				n2.x = normals[ base + j + 0 ];
				n2.y = normals[ base + j + 1 ];
				n2.z = normals[ base + j + 2 ];

				// average normals

				n.addVectors( n1, n2 ).normalize();

				// assign the new values to both normals

				normals[ j + 0 ] = normals[ base + j + 0 ] = n.x;
				normals[ j + 1 ] = normals[ base + j + 1 ] = n.y;
				normals[ j + 2 ] = normals[ base + j + 2 ] = n.z;

			}

		}
*/
	}

	static fromJSON( data ) {

		return new LatheGeometry( data.points, data.segments, data.phiStart, data.phiLength );

	}

}


export { LatheGeometry, LatheGeometry as LatheBufferGeometry };
