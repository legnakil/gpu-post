/*
	scrypt-jane by Andrew M, https://github.com/floodyberry/scrypt-jane

	OpenCL version by hanzac

	Support for LOOKUP_GAP and CONCURRENT_THREADS by mikaelh
	Nfactor compensation by mikaelh
	Keccak rewrite by mikaelh

	Public Domain or MIT License, whichever is easier
*/

#define SCRYPT_HASH "Keccak-512"
#define SCRYPT_HASH_DIGEST_SIZE 64
#define SCRYPT_KECCAK_F 1600
#define SCRYPT_HASH_BLOCK_SIZE 72
#define SCRYPT_BLOCK_BYTES 128
#define ROTL64(x, y) as_uint2(rotate(as_ulong(x), y))
#define ROTL32(x, y) rotate(x, y)

typedef struct scrypt_hash_state_t {
	uint4 state4[(SCRYPT_KECCAK_F + 127) / 128]; // 8 bytes of extra
	uint4 buffer4[(SCRYPT_HASH_BLOCK_SIZE + 15) / 16]; // 8 bytes of extra
	//uint leftover;
} scrypt_hash_state;

typedef struct scrypt_hmac_state_t {
	scrypt_hash_state inner;
	scrypt_hash_state outer;
} scrypt_hmac_state;


__constant ulong keccak_round_constants[24] = {
	0x0000000000000001UL, 0x0000000000008082UL,
	0x800000000000808aUL, 0x8000000080008000UL,
	0x000000000000808bUL, 0x0000000080000001UL,
	0x8000000080008081UL, 0x8000000000008009UL,
	0x000000000000008aUL, 0x0000000000000088UL,
	0x0000000080008009UL, 0x000000008000000aUL,
	0x000000008000808bUL, 0x800000000000008bUL,
	0x8000000000008089UL, 0x8000000000008003UL,
	0x8000000000008002UL, 0x8000000000000080UL,
	0x000000000000800aUL, 0x800000008000000aUL,
	0x8000000080008081UL, 0x8000000000008080UL,
	0x0000000080000001UL, 0x8000000080008008UL
};


static void
keccak_block_core(scrypt_hash_state *S) {
	uint2 t[5];
	uint2 u[5];
	uint2 v;
	uint2 w;
	uint4 *s4 = S->state4;
	uint i;
	
	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s4[0].xy ^ s4[2].zw ^ s4[5].xy ^ s4[7].zw ^ s4[10].xy;
		t[1] = s4[0].zw ^ s4[3].xy ^ s4[5].zw ^ s4[8].xy ^ s4[10].zw;
		t[2] = s4[1].xy ^ s4[3].zw ^ s4[6].xy ^ s4[8].zw ^ s4[11].xy;
		t[3] = s4[1].zw ^ s4[4].xy ^ s4[6].zw ^ s4[9].xy ^ s4[11].zw;
		t[4] = s4[2].xy ^ s4[4].zw ^ s4[7].xy ^ s4[9].zw ^ s4[12].xy;
	
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1UL);
		u[1] = t[0] ^ ROTL64(t[2], 1UL);
		u[2] = t[1] ^ ROTL64(t[3], 1UL);
		u[3] = t[2] ^ ROTL64(t[4], 1UL);
		u[4] = t[3] ^ ROTL64(t[0], 1UL);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s4[0].xy ^= u[0]; s4[2].zw ^= u[0]; s4[5].xy ^= u[0]; s4[7].zw ^= u[0]; s4[10].xy ^= u[0];
		s4[0].zw ^= u[1]; s4[3].xy ^= u[1]; s4[5].zw ^= u[1]; s4[8].xy ^= u[1]; s4[10].zw ^= u[1];
		s4[1].xy ^= u[2]; s4[3].zw ^= u[2]; s4[6].xy ^= u[2]; s4[8].zw ^= u[2]; s4[11].xy ^= u[2];
		s4[1].zw ^= u[3]; s4[4].xy ^= u[3]; s4[6].zw ^= u[3]; s4[9].xy ^= u[3]; s4[11].zw ^= u[3];
		s4[2].xy ^= u[4]; s4[4].zw ^= u[4]; s4[7].xy ^= u[4]; s4[9].zw ^= u[4]; s4[12].xy ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s4[0].zw;
		s4[ 0].zw = ROTL64(s4[ 3].xy, 44UL);
		s4[ 3].xy = ROTL64(s4[ 4].zw, 20UL);
		s4[ 4].zw = ROTL64(s4[11].xy, 61UL);
		s4[11].xy = ROTL64(s4[ 7].xy, 39UL);
		s4[ 7].xy = ROTL64(s4[10].xy, 18UL);
		s4[10].xy = ROTL64(s4[ 1].xy, 62UL);
		s4[ 1].xy = ROTL64(s4[ 6].xy, 43UL);
		s4[ 6].xy = ROTL64(s4[ 6].zw, 25UL);
		s4[ 6].zw = ROTL64(s4[ 9].zw,  8UL);
		s4[ 9].zw = ROTL64(s4[11].zw, 56UL);
		s4[11].zw = ROTL64(s4[ 7].zw, 41UL);
		s4[ 7].zw = ROTL64(s4[ 2].xy, 27UL);
		s4[ 2].xy = ROTL64(s4[12].xy, 14UL);
		s4[12].xy = ROTL64(s4[10].zw,  2UL);
		s4[10].zw = ROTL64(s4[ 4].xy, 55UL);
		s4[ 4].xy = ROTL64(s4[ 8].xy, 45UL);
		s4[ 8].xy = ROTL64(s4[ 2].zw, 36UL);
		s4[ 2].zw = ROTL64(s4[ 1].zw, 28UL);
		s4[ 1].zw = ROTL64(s4[ 9].xy, 21UL);
		s4[ 9].xy = ROTL64(s4[ 8].zw, 15UL);
		s4[ 8].zw = ROTL64(s4[ 5].zw, 10UL);
		s4[ 5].zw = ROTL64(s4[ 3].zw,  6UL);
		s4[ 3].zw = ROTL64(s4[ 5].xy,  3UL);
		s4[ 5].xy = ROTL64(        v,  1UL);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s4[ 0].xy; w = s4[ 0].zw; s4[ 0].xy ^= (~w) & s4[ 1].xy; s4[ 0].zw ^= (~s4[ 1].xy) & s4[ 1].zw; s4[ 1].xy ^= (~s4[ 1].zw) & s4[ 2].xy; s4[ 1].zw ^= (~s4[ 2].xy) & v; s4[ 2].xy ^= (~v) & w;
		v = s4[ 2].zw; w = s4[ 3].xy; s4[ 2].zw ^= (~w) & s4[ 3].zw; s4[ 3].xy ^= (~s4[ 3].zw) & s4[ 4].xy; s4[ 3].zw ^= (~s4[ 4].xy) & s4[ 4].zw; s4[ 4].xy ^= (~s4[ 4].zw) & v; s4[ 4].zw ^= (~v) & w;
		v = s4[ 5].xy; w = s4[ 5].zw; s4[ 5].xy ^= (~w) & s4[ 6].xy; s4[ 5].zw ^= (~s4[ 6].xy) & s4[ 6].zw; s4[ 6].xy ^= (~s4[ 6].zw) & s4[ 7].xy; s4[ 6].zw ^= (~s4[ 7].xy) & v; s4[ 7].xy ^= (~v) & w;
		v = s4[ 7].zw; w = s4[ 8].xy; s4[ 7].zw ^= (~w) & s4[ 8].zw; s4[ 8].xy ^= (~s4[ 8].zw) & s4[ 9].xy; s4[ 8].zw ^= (~s4[ 9].xy) & s4[ 9].zw; s4[ 9].xy ^= (~s4[ 9].zw) & v; s4[ 9].zw ^= (~v) & w;
		v = s4[10].xy; w = s4[10].zw; s4[10].xy ^= (~w) & s4[11].xy; s4[10].zw ^= (~s4[11].xy) & s4[11].zw; s4[11].xy ^= (~s4[11].zw) & s4[12].xy; s4[11].zw ^= (~s4[12].xy) & v; s4[12].xy ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s4[0].xy ^= as_uint2(keccak_round_constants[i]);
	}
}

__constant uint4 ZERO = (uint4)(0);
__constant uint2 ZERO_UINT2 = (uint2)(0);

static void
keccak_block(scrypt_hash_state *S, const uint4 *in4) {
	uint4 *s4 = S->state4;
	uint i;

	/* absorb input */
	#pragma unroll
	for (i = 0; i < 4; i++) {
		s4[i] ^= in4[i];
	}
	s4[4].xy ^= in4[4].xy;
	
	keccak_block_core(S);
}

static void
keccak_block_zero(scrypt_hash_state *S, const uint4 *in4) {
	uint4 *s4 = S->state4;
	uint i;
	
	/* absorb input */
	#pragma unroll
	for (i = 0; i < 4; i++) {
		s4[i] = in4[i];
	}
	s4[4].xyzw = (uint4)(in4[4].xy, 0, 0);
	
	#pragma unroll
	for (i = 5; i < 12; i++) {
		s4[i] = ZERO;
	}
	s4[12].xy = ZERO_UINT2;
	
	keccak_block_core(S);
}

static void
scrypt_hash_update_72(scrypt_hash_state *S, const uint4 *in4) {
	/* handle the current data */
	keccak_block_zero(S, in4);
}

static void
scrypt_hash_update_80(scrypt_hash_state *S, const uint4 *in4) {
	const uchar *in = (const uchar *)in4;
	uint i;

	/* handle the current data */
	keccak_block(S, in4);
	in += SCRYPT_HASH_BLOCK_SIZE;

	/* handle leftover data */
	//S->leftover = 2;
	
	{
		const uint2 *int2 = (const uint2 *) in;

		S->buffer4[0].xy = int2[0].xy;
	}
}

static void
scrypt_hash_update_128(scrypt_hash_state *S, const uint4 *in4) {
	const uchar *in = (const uchar *)in4;
	uint i;

	/* handle the current data */
	keccak_block(S, in4);
	in += SCRYPT_HASH_BLOCK_SIZE;

	/* handle leftover data */
	//S->leftover = 14;
	
	{
		const uint2 *int2 = (const uint2 *) in;
		
		#pragma unroll
		for (i = 0; i < 3; i++) {
			S->buffer4[i] = (uint4)(int2[i<<1].xy, int2[(i<<1) + 1].xy);
		}
		S->buffer4[3].xy = int2[6].xy;
	}
}

static void
scrypt_hash_update_4_after_72(scrypt_hash_state *S, uint in) {
	// assume that leftover = 0
	/* handle the previous data */
	S->buffer4[0] = (uint4)(in, 0x01, 0, 0);
	//S->leftover += 1;
}

static void
scrypt_hash_update_4_after_80(scrypt_hash_state *S, uint in) {
	// assume that leftover = 2
	/* handle the previous data */
	S->buffer4[0].zw = (uint2)(in, 0x01);
	//S->leftover += 1;
}

static void
scrypt_hash_update_4_after_128(scrypt_hash_state *S, uint in) {
	// leftover = 14
	/* handle the previous data */
	S->buffer4[3].zw = (uint2)(in, 0x01);
	//S->leftover += 1;
}

static void
scrypt_hash_update_64(scrypt_hash_state *S, const uint4 *in4) {
	uint i;

	/* handle leftover data */
	//S->leftover = 16;

	#pragma unroll
	for (i = 0; i < 4; i++) {
		S->buffer4[i] = in4[i];
	}
}

static void
scrypt_hash_finish_80_after_64(scrypt_hash_state *S, uint4 *hash4) {
	// assume that leftover = 16
	S->buffer4[4].xy = (uint2)(0x01, 0x80000000);
	
	keccak_block(S, S->buffer4);
	
	#pragma unroll
	for (uint i = 0; i < 4; i++) {
		hash4[i] = S->state4[i];
	}
}

static uchar
scrypt_hash_finish_80_after_64_1(scrypt_hash_state *S) {
	// assume that leftover = 16
	S->buffer4[4].xy = (uint2)(0x01, 0x80000000);
	
	keccak_block(S, S->buffer4);

	return S->state4[0].x;
}

static void
scrypt_hash_finish_80_after_72_4(scrypt_hash_state *S, uint4 *hash4) {
	uint i;
	// assume that leftover = 1
	//S->buffer4[0].y = 0x01; // done already in scrypt_hash_update_4_after_80
	#pragma unroll
	for (i = 1; i < 4; i++) {
		S->buffer4[i] = ZERO;
	}
	S->buffer4[4].xy = (uint2)(0, 0x80000000);
	
	keccak_block(S, S->buffer4);
	
	#pragma unroll
	for (uint i = 0; i < 4; i++) {
		hash4[i] = S->state4[i];
	}
}

static void
scrypt_hash_finish_80_after_80_4(scrypt_hash_state *S, uint4 *hash4) {
	uint i;
	// assume that leftover = 3
	//S->buffer4[0].w = 0x01; // done already in scrypt_hash_update_4_after_80
	#pragma unroll
	for (i = 1; i < 4; i++) {
		S->buffer4[i] = ZERO;
	}
	S->buffer4[4].xy = (uint2)(0, 0x80000000);
	
	keccak_block(S, S->buffer4);
	
	#pragma unroll
	for (uint i = 0; i < 4; i++) {
		hash4[i] = S->state4[i];
	}
}

static void
scrypt_hash_finish_80_after_128_4(scrypt_hash_state *S, uint4 *hash4) {
	// leftover = 15
	//S->buffer4[3].w = 0x01; // done already in scrypt_hash_update_4_after_128
	S->buffer4[4].xy = (uint2)(0, 0x80000000);
	
	keccak_block(S, S->buffer4);
	
	#pragma unroll
	for (uint i = 0; i < 4; i++) {
		hash4[i] = S->state4[i];
	}
}

static void
scrypt_hash_80(uint4 *hash4, const uint4 *m) {
	const uchar *in = (const uchar *)m;
	scrypt_hash_state st;
	uint i;
	
	/* handle the current data */
	keccak_block_zero(&st, m);
	in += SCRYPT_HASH_BLOCK_SIZE;

	{
		const uint2 *in2 = (const uint2 *) in;
		st.buffer4[0].xyzw = (uint4)(in2[0].xy, 0x01, 0);
	}

	#pragma unroll
	for (i = 1; i < 4; i++) {
		st.buffer4[i] = ZERO;
	}
	st.buffer4[4].xyzw = (uint4)(0, 0x80000000, 0, 0);

	keccak_block(&st, st.buffer4);

	#pragma unroll
	for (i = 0; i < 4; i++) {
		hash4[i] = st.state4[i];
	}
}

/* hmac */
__constant uint4 KEY_0X36 = (uint4)(0x36363636);
__constant uint2 KEY_0X36_2 = (uint2)(0x36363636);
__constant uint4 KEY_0X36_XOR_0X5C = (uint4)(0x6A6A6A6A);
__constant uint2 KEY_0X36_XOR_0X5C_2 = (uint2)(0x6A6A6A6A);

static void
scrypt_hmac_init(scrypt_hmac_state *st, const uint4 *key) {
	uint4 pad4[SCRYPT_HASH_BLOCK_SIZE/16 + 1];
	uint i;

	/* if it's > blocksize bytes, hash it */
	#pragma unroll
	for (i = 0; i < 4; i++) {
		pad4[i] = key[i];
	}
	pad4[4].xy = key[4].xy;

	/* inner = (key ^ 0x36) */
	/* h(inner || ...) */
	#pragma unroll
	for (i = 0; i < 4; i++) {
		pad4[i] ^= KEY_0X36;
	}
	pad4[4].xy ^= KEY_0X36_2;
	scrypt_hash_update_72(&st->inner, pad4);

	/* outer = (key ^ 0x5c) */
	/* h(outer || ...) */
	#pragma unroll
	for (i = 0; i < 4; i++) {
		pad4[i] ^= KEY_0X36_XOR_0X5C;
	}
	pad4[4].xy ^= KEY_0X36_XOR_0X5C_2;
	scrypt_hash_update_72(&st->outer, pad4);
}

static void
scrypt_hmac_update_80(scrypt_hmac_state *st, const uint4 *m) {
	/* h(inner || m...) */
	scrypt_hash_update_80(&st->inner, m);
}

static void
scrypt_hmac_update_128(scrypt_hmac_state *st, const uint4 *m) {
	/* h(inner || m...) */
	scrypt_hash_update_128(&st->inner, m);
}

static void
scrypt_hmac_update_4_after_72(scrypt_hmac_state *st, uint m) {
	/* h(inner || m...) */
	scrypt_hash_update_4_after_72(&st->inner, m);
}

static void
scrypt_hmac_update_4_after_80(scrypt_hmac_state *st, uint m) {
	/* h(inner || m...) */
	scrypt_hash_update_4_after_80(&st->inner, m);
}

static void
scrypt_hmac_update_4_after_128(scrypt_hmac_state *st, uint m) {
	/* h(inner || m...) */
	scrypt_hash_update_4_after_128(&st->inner, m);
}

static void
scrypt_hmac_finish_128B(scrypt_hmac_state *st, uint4 *mac) {
	/* h(inner || m) */
	uint4 innerhash[4];
	scrypt_hash_finish_80_after_72_4(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	scrypt_hash_update_64(&st->outer, innerhash);
	scrypt_hash_finish_80_after_64(&st->outer, mac);
}

static uchar
scrypt_hmac_finish_32B(scrypt_hmac_state *st) {
	/* h(inner || m) */
	uint4 innerhash[4];
	scrypt_hash_finish_80_after_128_4(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	scrypt_hash_update_64(&st->outer, innerhash);
	return scrypt_hash_finish_80_after_64_1(&st->outer);
}

static void
scrypt_copy_hmac_state_128B(scrypt_hmac_state *dest, const scrypt_hmac_state *src) {
	uint i;

	for (i = 0; i < 12; i++) {
		dest->inner.state4[i] = src->inner.state4[i];
	}
	dest->inner.state4[12].xy = src->inner.state4[12].xy;

	dest->inner.buffer4[0].xy = src->inner.buffer4[0].xy;

	for (i = 0; i < 12; i++) {
		dest->outer.state4[i] = src->outer.state4[i];
	}
	dest->outer.state4[12].xy = src->outer.state4[12].xy;
}

__constant uint be1 = 0x01000000;
__constant uint be2 = 0x02000000;

static void
scrypt_pbkdf2_128B(const uint4 *password, uint4 *out4) {
	scrypt_hmac_state hmac_pw, work;
	uint4 ti4[4];
	uint i;
	
	/* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they will always be under scrypt */

	/* hmac(password, ...) */
	scrypt_hmac_init(&hmac_pw, password);

	/* U1 = hmac(password, salt || be(i)) */
	/* U32TO8_BE(be, i); */
	//work = hmac_pw;
	scrypt_copy_hmac_state_128B(&work, &hmac_pw);
	scrypt_hmac_update_4_after_72(&work, be1);
	scrypt_hmac_finish_128B(&work, ti4);

	#pragma unroll
	for (i = 0; i < 4; i++) {
		out4[i] = ti4[i];
	}
		
	/* U1 = hmac(password, salt || be(i)) */
	/* U32TO8_BE(be, i); */
	// work = hmac_pw;
	scrypt_hmac_update_4_after_72(&hmac_pw, be2);
	scrypt_hmac_finish_128B(&hmac_pw, ti4);

	#pragma unroll
	for (i = 0; i < 4; i++) {
		out4[i + 4] = ti4[i];
	}
}

static uchar
scrypt_pbkdf2_32B(const uint4 *password, const uint4 *salt) {
	scrypt_hmac_state hmac_pw;
	
	/* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they will always be under scrypt */

	/* hmac(password, ...) */
	scrypt_hmac_init(&hmac_pw, password);

	/* hmac(password, salt...) */
	scrypt_hmac_update_128(&hmac_pw, salt);

	/* U1 = hmac(password, salt || be(i)) */
	/* U32TO8_BE(be, i); */
	scrypt_hmac_update_4_after_128(&hmac_pw, be1);
	return scrypt_hmac_finish_32B(&hmac_pw);
}

__constant uint4 MASK_2 = (uint4) (1, 2, 3, 0);
__constant uint4 MASK_3 = (uint4) (2, 3, 0, 1);
__constant uint4 MASK_4 = (uint4) (3, 0, 1, 2);
__constant uint4 ROTATE_16 = (uint4) (16, 16, 16, 16);
__constant uint4 ROTATE_12 = (uint4) (12, 12, 12, 12);
__constant uint4 ROTATE_8 = (uint4) (8, 8, 8, 8);
__constant uint4 ROTATE_7 = (uint4) (7, 7, 7, 7);

static void
chacha_core(uint4 state[4]) {
	uint4 x[4];
	uint4 t;
	uint rounds;

	x[0] = state[0];
	x[1] = state[1];
	x[2] = state[2];
	x[3] = state[3];

	#pragma unroll
	for (rounds = 0; rounds < 4; rounds ++) {
		x[0] += x[1]; t = x[3] ^ x[0]; x[3] = ROTL32(t, ROTATE_16);
		x[2] += x[3]; t = x[1] ^ x[2]; x[1] = ROTL32(t, ROTATE_12);
		x[0] += x[1]; t = x[3] ^ x[0]; x[3] = ROTL32(t, ROTATE_8);
		x[2] += x[3]; t = x[1] ^ x[2]; x[1] = ROTL32(t, ROTATE_7);
		
		// x[1] = shuffle(x[1], MASK_2);
		// x[2] = shuffle(x[2], MASK_3);
		// x[3] = shuffle(x[3], MASK_4);
		
		x[0]      += x[1].yzwx; t = x[3].wxyz ^ x[0];      x[3].wxyz = ROTL32(t, ROTATE_16);
		x[2].zwxy += x[3].wxyz; t = x[1].yzwx ^ x[2].zwxy; x[1].yzwx = ROTL32(t, ROTATE_12);
		x[0]      += x[1].yzwx; t = x[3].wxyz ^ x[0];      x[3].wxyz = ROTL32(t, ROTATE_8);
		x[2].zwxy += x[3].wxyz; t = x[1].yzwx ^ x[2].zwxy; x[1].yzwx = ROTL32(t, ROTATE_7);
		
		// x[1] = shuffle(x[1], MASK_4);
		// x[2] = shuffle(x[2], MASK_3);
		// x[3] = shuffle(x[3], MASK_2);
	}

	state[0] += x[0];
	state[1] += x[1];
	state[2] += x[2];
	state[3] += x[3];
}

static void
scrypt_ChunkMix_inplace_Bxor_local(uint4 *restrict B/*[chunkWords]*/, uint4 *restrict Bxor/*[chunkWords]*/) {
	/* 1: X = B_{2r - 1} */

	/* 2: for i = 0 to 2r - 1 do */
		/* 3: X = H(X ^ B_i) */
		B[0] ^= B[4] ^ Bxor[4] ^ Bxor[0];
		B[1] ^= B[5] ^ Bxor[5] ^ Bxor[1];
		B[2] ^= B[6] ^ Bxor[6] ^ Bxor[2];
		B[3] ^= B[7] ^ Bxor[7] ^ Bxor[3];
		
		/* SCRYPT_MIX_FN */ chacha_core(B);

		/* 4: Y_i = X */
		/* 6: B'[0..r-1] = Y_even */
		/* 6: B'[r..2r-1] = Y_odd */


		/* 3: X = H(X ^ B_i) */
		B[4] ^= B[0] ^ Bxor[4];
		B[5] ^= B[1] ^ Bxor[5];
		B[6] ^= B[2] ^ Bxor[6];
		B[7] ^= B[3] ^ Bxor[7];
		
		/* SCRYPT_MIX_FN */ chacha_core(B + 4);

		/* 4: Y_i = X */
		/* 6: B'[0..r-1] = Y_even */
		/* 6: B'[r..2r-1] = Y_odd */
}

static void
scrypt_ChunkMix_inplace_local(uint4 *restrict B/*[chunkWords]*/) {
	/* 1: X = B_{2r - 1} */

	/* 2: for i = 0 to 2r - 1 do */
		/* 3: X = H(X ^ B_i) */
		B[0] ^= B[4];
		B[1] ^= B[5];
		B[2] ^= B[6];
		B[3] ^= B[7];

		/* SCRYPT_MIX_FN */ chacha_core(B);

		/* 4: Y_i = X */
		/* 6: B'[0..r-1] = Y_even */
		/* 6: B'[r..2r-1] = Y_odd */


		/* 3: X = H(X ^ B_i) */
		B[4] ^= B[0];
		B[5] ^= B[1];
		B[6] ^= B[2];
		B[7] ^= B[3];

		/* SCRYPT_MIX_FN */ chacha_core(B + 4);

		/* 4: Y_i = X */
		/* 6: B'[0..r-1] = Y_even */
		/* 6: B'[r..2r-1] = Y_odd */
}

#define Coord(x,y,z) x+y*(x ## SIZE)+z*(y ## SIZE)*(x ## SIZE)
#define CO Coord(z,x,y)

static void
scrypt_ROMix(uint4 *restrict X/*[chunkWords]*/, __global uint4 *restrict lookup/*[N * chunkWords]*/, const uint N, const uint gid, const uint Nfactor) {
	const uint effective_concurrency = (CONCURRENT_THREADS << 9) >> Nfactor;
	const uint zSIZE = 8;
	const uint ySIZE = (N/LOOKUP_GAP+(N%LOOKUP_GAP>0));
	const uint xSIZE = effective_concurrency;
	const uint x = gid % xSIZE;
	uint i, j, y, z;
	uint4 W[8];

	/* 1: X = B */
	/* implicit */

	/* 2: for i = 0 to N - 1 do */
	for (y = 0; y < N / LOOKUP_GAP; y++) {
		/* 3: V_i = X */
		#pragma unroll
		for (z = 0; z < zSIZE; z++) {
			lookup[CO] = X[z];
		}

		for (j = 0; j < LOOKUP_GAP; j++) {
			/* 4: X = H(X) */
			scrypt_ChunkMix_inplace_local(X);
		}
	}

#if (LOOKUP_GAP != 1) && (LOOKUP_GAP != 2) && (LOOKUP_GAP != 4) && (LOOKUP_GAP != 8)
	if (N % LOOKUP_GAP > 0) {
		y = N / LOOKUP_GAP;

		#pragma unroll
		for (z = 0; z < zSIZE; z++) {
			lookup[CO] = X[z];
		}

		for (j = 0; j < N % LOOKUP_GAP; j++) {
			scrypt_ChunkMix_inplace_local(X);
		}
	}
#endif

	/* 6: for i = 0 to N - 1 do */
	for (i = 0; i < N; i++) {
		/* 7: j = Integerify(X) % N */
		j = X[4].x & (N - 1);
		y = j / LOOKUP_GAP;

		#pragma unroll
		for (z = 0; z < zSIZE; z++) {
			W[z] = lookup[CO];
		}

		
#if (LOOKUP_GAP == 1)
#elif (LOOKUP_GAP == 2)
		if (j & 1) {
			scrypt_ChunkMix_inplace_local(W);
		}
#else
		uint c = j % LOOKUP_GAP;
		for (uint k = 0; k < c; k++) {
			scrypt_ChunkMix_inplace_local(W);
		}
#endif

		/* 8: X = H(X ^ V_j) */
		scrypt_ChunkMix_inplace_Bxor_local(X, W);
	}

	/* 10: B' = X */
	/* implicit */
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search8(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X);

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint8 buffer;

		output += gid * 64 / 64;

		buffer.s0  = labels[0];
		buffer.s0 |= labels[1] << 8;
		buffer.s0 |= labels[2] << 16;
		buffer.s0 |= labels[3] << 24;

		buffer.s1  = labels[4];
		buffer.s1 |= labels[5] << 8;
		buffer.s1 |= labels[6] << 16;
		buffer.s1 |= labels[7] << 24;

		buffer.s2  = labels[8];
		buffer.s2 |= labels[9] << 8;
		buffer.s2 |= labels[10] << 16;
		buffer.s2 |= labels[11] << 24;

		buffer.s3  = labels[12];
		buffer.s3 |= labels[13] << 8;
		buffer.s3 |= labels[14] << 16;
		buffer.s3 |= labels[15] << 24;

		buffer.s4  = labels[16];
		buffer.s4 |= labels[17] << 8;
		buffer.s4 |= labels[18] << 16;
		buffer.s4 |= labels[19] << 24;

		buffer.s5  = labels[20];
		buffer.s5 |= labels[21] << 8;
		buffer.s5 |= labels[22] << 16;
		buffer.s5 |= labels[23] << 24;

		buffer.s6  = labels[24];
		buffer.s6 |= labels[25] << 8;
		buffer.s6 |= labels[26] << 16;
		buffer.s6 |= labels[27] << 24;

		buffer.s7  = labels[28];
		buffer.s7 |= labels[29] << 8;
		buffer.s7 |= labels[30] << 16;
		buffer.s7 |= labels[31] << 24;

		*(__global uint8*)output = buffer;
		output += 32;

		buffer.s0  = labels[32];
		buffer.s0 |= labels[33] << 8;
		buffer.s0 |= labels[34] << 16;
		buffer.s0 |= labels[35] << 24;

		buffer.s1  = labels[36];
		buffer.s1 |= labels[37] << 8;
		buffer.s1 |= labels[38] << 16;
		buffer.s1 |= labels[39] << 24;

		buffer.s2  = labels[40];
		buffer.s2 |= labels[41] << 8;
		buffer.s2 |= labels[42] << 16;
		buffer.s2 |= labels[43] << 24;

		buffer.s3  = labels[44];
		buffer.s3 |= labels[45] << 8;
		buffer.s3 |= labels[46] << 16;
		buffer.s3 |= labels[47] << 24;

		buffer.s4  = labels[48];
		buffer.s4 |= labels[49] << 8;
		buffer.s4 |= labels[50] << 16;
		buffer.s4 |= labels[51] << 24;

		buffer.s5  = labels[52];
		buffer.s5 |= labels[53] << 8;
		buffer.s5 |= labels[54] << 16;
		buffer.s5 |= labels[55] << 24;

		buffer.s6  = labels[56];
		buffer.s6 |= labels[57] << 8;
		buffer.s6 |= labels[58] << 16;
		buffer.s6 |= labels[59] << 24;

		buffer.s7  = labels[60];
		buffer.s7 |= labels[61] << 8;
		buffer.s7 |= labels[62] << 16;
		buffer.s7 |= labels[63] << 24;

		*(__global uint8*)output = buffer;
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search7(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X) & 0x7f;

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint2 buffer1, buffer2, buffer3, buffer4, buffer5, buffer6, buffer7;

		output += gid * 56 / 64;

		buffer1.s0  = labels[0];
		buffer1.s0 |= labels[1] << 7;
		buffer1.s0 |= labels[2] << 14;
		buffer1.s0 |= labels[3] << 21;
		buffer1.s1  = labels[4];
		buffer1.s0 |= buffer1.s1 << 28;

		buffer1.s1 >>= 4;
		buffer1.s1 |= labels[5] << 3;
		buffer1.s1 |= labels[6] << 10;
		buffer1.s1 |= labels[7] << 17;
		buffer1.s1 |= labels[8] << 24;
		buffer2.s0  = labels[9];
		buffer1.s1 |= buffer2.s0 << 31;

		buffer2.s0 >>= 1;
		buffer2.s0 |= labels[10] << 6;
		buffer2.s0 |= labels[11] << 13;
		buffer2.s0 |= labels[12] << 20;
		buffer2.s1  = labels[13];
		buffer2.s0 |= buffer2.s1 << 27;

		buffer2.s1 >>= 5;
		buffer2.s1 |= labels[14] << 2;
		buffer2.s1 |= labels[15] << 9;
		buffer2.s1 |= labels[16] << 16;
		buffer2.s1 |= labels[17] << 23;
		buffer3.s0  = labels[18];
		buffer2.s1 |= buffer3.s0 << 30;

		buffer3.s0 >>= 2;
		buffer3.s0 |= labels[19] << 5;
		buffer3.s0 |= labels[20] << 12;
		buffer3.s0 |= labels[21] << 19;
		buffer3.s1  = labels[22];
		buffer3.s0 |= buffer3.s1 << 26;

		buffer3.s1 >>= 6;
		buffer3.s1 |= labels[23] << 1;
		buffer3.s1 |= labels[24] << 8;
		buffer3.s1 |= labels[25] << 15;
		buffer3.s1 |= labels[26] << 22;
		buffer4.s0  = labels[27];
		buffer3.s1 |= buffer4.s0 << 29;

		buffer4.s0 >>= 3;
		buffer4.s0 |= labels[28] << 4;
		buffer4.s0 |= labels[29] << 11;
		buffer4.s0 |= labels[30] << 18;
		buffer4.s0 |= labels[31] << 25;
//
		buffer4.s1  = labels[32+0];
		buffer4.s1 |= labels[32+1] << 7;
		buffer4.s1 |= labels[32+2] << 14;
		buffer4.s1 |= labels[32+3] << 21;
		buffer5.s0  = labels[32+4];
		buffer4.s1 |= buffer5.s0 << 28;

		buffer5.s0 >>= 4;
		buffer5.s0 |= labels[32+5] << 3;
		buffer5.s0 |= labels[32+6] << 10;
		buffer5.s0 |= labels[32+7] << 17;
		buffer5.s0 |= labels[32+8] << 24;
		buffer5.s1  = labels[32+9];
		buffer5.s0 |= buffer5.s1 << 31;

		buffer5.s1 >>= 1;
		buffer5.s1 |= labels[32+10] << 6;
		buffer5.s1 |= labels[32+11] << 13;
		buffer5.s1 |= labels[32+12] << 20;
		buffer6.s0  = labels[32+13];
		buffer5.s1 |= buffer6.s0 << 27;

		buffer6.s0 >>= 5;
		buffer6.s0 |= labels[32+14] << 2;
		buffer6.s0 |= labels[32+15] << 9;
		buffer6.s0 |= labels[32+16] << 16;
		buffer6.s0 |= labels[32+17] << 23;
		buffer6.s1  = labels[32+18];
		buffer6.s0 |= buffer6.s1 << 30;

		buffer6.s1 >>= 2;
		buffer6.s1 |= labels[32+19] << 5;
		buffer6.s1 |= labels[32+20] << 12;
		buffer6.s1 |= labels[32+21] << 19;
		buffer7.s0  = labels[32+22];
		buffer6.s1 |= buffer7.s0 << 26;

		buffer7.s0 >>= 6;
		buffer7.s0 |= labels[32+23] << 1;
		buffer7.s0 |= labels[32+24] << 8;
		buffer7.s0 |= labels[32+25] << 15;
		buffer7.s0 |= labels[32+26] << 22;
		buffer7.s1  = labels[32+27];
		buffer7.s0 |= buffer7.s1 << 29;

		buffer7.s1 >>= 3;
		buffer7.s1 |= labels[32+28] << 4;
		buffer7.s1 |= labels[32+29] << 11;
		buffer7.s1 |= labels[32+30] << 18;
		buffer7.s1 |= labels[32+31] << 25;

		((__global uint2*)output)[0] = buffer1;
		((__global uint2*)output)[1] = buffer2;
		((__global uint2*)output)[2] = buffer3;
		((__global uint2*)output)[3] = buffer4;
		((__global uint2*)output)[4] = buffer5;
		((__global uint2*)output)[5] = buffer6;
		((__global uint2*)output)[6] = buffer7;
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search6(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X) & 0x3f;

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint4 buffer1, buffer2, buffer3;

		output += gid * 48 / 64;

		buffer1.s0  = labels[0*16+0];
		buffer1.s0 |= labels[0*16+1] << 6;
		buffer1.s0 |= labels[0*16+2] << 12;
		buffer1.s0 |= labels[0*16+3] << 18;
		buffer1.s0 |= labels[0*16+4] << 24;
		buffer1.s1  = labels[0*16+5];
		buffer1.s0 |= buffer1.s1 << 30;

		buffer1.s1 >>= 2;
		buffer1.s1 |= labels[0*16+6] << 4;
		buffer1.s1 |= labels[0*16+7] << 10;
		buffer1.s1 |= labels[0*16+8] << 16;
		buffer1.s1 |= labels[0*16+9] << 22;
		buffer1.s2  = labels[0*16+10];
		buffer1.s1 |= buffer1.s2 << 28;

		buffer1.s2 >>= 4;
		buffer1.s2 |= labels[0*16+11] << 2;
		buffer1.s2 |= labels[0*16+12] << 8;
		buffer1.s2 |= labels[0*16+13] << 14;
		buffer1.s2 |= labels[0*16+14] << 20;
		buffer1.s2 |= labels[0*16+15] << 26;

		buffer1.s3  = labels[1*16+0];
		buffer1.s3 |= labels[1*16+1] << 6;
		buffer1.s3 |= labels[1*16+2] << 12;
		buffer1.s3 |= labels[1*16+3] << 18;
		buffer1.s3 |= labels[1*16+4] << 24;
		buffer2.s0  = labels[1*16+5];
		buffer1.s3 |= buffer2.s0 << 30;

		buffer2.s0 >>= 2;
		buffer2.s0 |= labels[1*16+6] << 4;
		buffer2.s0 |= labels[1*16+7] << 10;
		buffer2.s0 |= labels[1*16+8] << 16;
		buffer2.s0 |= labels[1*16+9] << 22;
		buffer2.s1  = labels[1*16+10];
		buffer2.s0 |= buffer2.s1 << 28;

		buffer2.s1 >>= 4;
		buffer2.s1 |= labels[1*16+11] << 2;
		buffer2.s1 |= labels[1*16+12] << 8;
		buffer2.s1 |= labels[1*16+13] << 14;
		buffer2.s1 |= labels[1*16+14] << 20;
		buffer2.s1 |= labels[1*16+15] << 26;

		buffer2.s2  = labels[2*16+0];
		buffer2.s2 |= labels[2*16+1] << 6;
		buffer2.s2 |= labels[2*16+2] << 12;
		buffer2.s2 |= labels[2*16+3] << 18;
		buffer2.s2 |= labels[2*16+4] << 24;
		buffer2.s3  = labels[2*16+5];
		buffer2.s2 |= buffer2.s3 << 30;

		buffer2.s3 >>= 2;
		buffer2.s3 |= labels[2*16+6] << 4;
		buffer2.s3 |= labels[2*16+7] << 10;
		buffer2.s3 |= labels[2*16+8] << 16;
		buffer2.s3 |= labels[2*16+9] << 22;
		buffer3.s0  = labels[2*16+10];
		buffer2.s3 |= buffer3.s0 << 28;

		buffer3.s0 >>= 4;
		buffer3.s0 |= labels[2*16+11] << 2;
		buffer3.s0 |= labels[2*16+12] << 8;
		buffer3.s0 |= labels[2*16+13] << 14;
		buffer3.s0 |= labels[2*16+14] << 20;
		buffer3.s0 |= labels[2*16+15] << 26;

		buffer3.s1  = labels[3*16+0];
		buffer3.s1 |= labels[3*16+1] << 6;
		buffer3.s1 |= labels[3*16+2] << 12;
		buffer3.s1 |= labels[3*16+3] << 18;
		buffer3.s1 |= labels[3*16+4] << 24;
		buffer3.s2  = labels[3*16+5];
		buffer3.s1 |= buffer3.s2 << 30;

		buffer3.s2 >>= 2;
		buffer3.s2 |= labels[3*16+6] << 4;
		buffer3.s2 |= labels[3*16+7] << 10;
		buffer3.s2 |= labels[3*16+8] << 16;
		buffer3.s2 |= labels[3*16+9] << 22;
		buffer3.s3  = labels[3*16+10];
		buffer3.s2 |= buffer3.s3 << 28;

		buffer3.s3 >>= 4;
		buffer3.s3 |= labels[3*16+11] << 2;
		buffer3.s3 |= labels[3*16+12] << 8;
		buffer3.s3 |= labels[3*16+13] << 14;
		buffer3.s3 |= labels[3*16+14] << 20;
		buffer3.s3 |= labels[3*16+15] << 26;

		((__global uint4*)output)[0] = buffer1;
		((__global uint4*)output)[1] = buffer2;
		((__global uint4*)output)[2] = buffer3;
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search5(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X) & 0x1f;

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint2 buffer1, buffer2, buffer3, buffer4, buffer5;

		output += gid * 40 / 64;

		buffer1.s0  = labels[0];
		buffer1.s0 |= labels[1] << 5;
		buffer1.s0 |= labels[2] << 10;
		buffer1.s0 |= labels[3] << 15;
		buffer1.s0 |= labels[4] << 20;
		buffer1.s0 |= labels[5] << 25;
		buffer1.s1  = labels[6];
		buffer1.s0 |= buffer1.s1 << 30;

		buffer1.s1 >>= 2;
		buffer1.s1 |= labels[7] << 3;
		buffer1.s1 |= labels[8] << 8;
		buffer1.s1 |= labels[9] << 13;
		buffer1.s1 |= labels[10] << 18;
		buffer1.s1 |= labels[11] << 23;
		buffer2.s0  = labels[12];
		buffer1.s1 |= buffer2.s0 << 28;

		buffer2.s0 >>= 4;
		buffer2.s0 |= labels[13] << 1;
		buffer2.s0 |= labels[14] << 6;
		buffer2.s0 |= labels[15] << 11;
		buffer2.s0 |= labels[16] << 16;
		buffer2.s0 |= labels[17] << 21;
		buffer2.s0 |= labels[18] << 26;
		buffer2.s1  = labels[19];
		buffer2.s0 |= buffer2.s1 << 31;

		buffer2.s1 >>= 1;
		buffer2.s1 |= labels[20] << 4;
		buffer2.s1 |= labels[21] << 9;
		buffer2.s1 |= labels[22] << 14;
		buffer2.s1 |= labels[23] << 19;
		buffer2.s1 |= labels[24] << 24;
		buffer3.s0  = labels[25];
		buffer2.s1 |= buffer3.s0 << 29;

		buffer3.s0 >>= 3;
		buffer3.s0 |= labels[26] << 2;
		buffer3.s0 |= labels[27] << 7;
		buffer3.s0 |= labels[28] << 12;
		buffer3.s0 |= labels[29] << 17;
		buffer3.s0 |= labels[30] << 22;
		buffer3.s0 |= labels[31] << 27;
//
		buffer3.s1  = labels[32+0];
		buffer3.s1 |= labels[32+1] << 5;
		buffer3.s1 |= labels[32+2] << 10;
		buffer3.s1 |= labels[32+3] << 15;
		buffer3.s1 |= labels[32+4] << 20;
		buffer3.s1 |= labels[32+5] << 25;
		buffer4.s0  = labels[32+6];
		buffer3.s1 |= buffer4.s0 << 30;

		buffer4.s0 >>= 2;
		buffer4.s0 |= labels[32+7] << 3;
		buffer4.s0 |= labels[32+8] << 8;
		buffer4.s0 |= labels[32+9] << 13;
		buffer4.s0 |= labels[32+10] << 18;
		buffer4.s0 |= labels[32+11] << 23;
		buffer4.s1  = labels[32+12];
		buffer4.s0 |= buffer4.s1 << 28;

		buffer4.s1 >>= 4;
		buffer4.s1 |= labels[32+13] << 1;
		buffer4.s1 |= labels[32+14] << 6;
		buffer4.s1 |= labels[32+15] << 11;
		buffer4.s1 |= labels[32+16] << 16;
		buffer4.s1 |= labels[32+17] << 21;
		buffer4.s1 |= labels[32+18] << 26;
		buffer5.s0  = labels[32+19];
		buffer4.s1 |= buffer5.s0 << 31;

		buffer5.s0 >>= 1;
		buffer5.s0 |= labels[32+20] << 4;
		buffer5.s0 |= labels[32+21] << 9;
		buffer5.s0 |= labels[32+22] << 14;
		buffer5.s0 |= labels[32+23] << 19;
		buffer5.s0 |= labels[32+24] << 24;
		buffer5.s1  = labels[32+25];
		buffer5.s0 |= buffer5.s1 << 29;

		buffer5.s1 >>= 3;
		buffer5.s1 |= labels[32+26] << 2;
		buffer5.s1 |= labels[32+27] << 7;
		buffer5.s1 |= labels[32+28] << 12;
		buffer5.s1 |= labels[32+29] << 17;
		buffer5.s1 |= labels[32+30] << 22;
		buffer5.s1 |= labels[32+31] << 27;

		((__global uint2*)output)[0] = buffer1;
		((__global uint2*)output)[1] = buffer2;
		((__global uint2*)output)[2] = buffer3;
		((__global uint2*)output)[3] = buffer4;
		((__global uint2*)output)[4] = buffer5;
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search4(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X) & 0x0f;

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint8 buffer;

		output += gid * 32 / 64;

		buffer.s0  = labels[0];
		buffer.s0 |= labels[1] << 4;
		buffer.s0 |= labels[2] << 8;
		buffer.s0 |= labels[3] << 12;
		buffer.s0 |= labels[4] << 16;
		buffer.s0 |= labels[5] << 20;
		buffer.s0 |= labels[6] << 24;
		buffer.s0 |= labels[7] << 28;

		buffer.s1  = labels[8];
		buffer.s1 |= labels[9] << 4;
		buffer.s1 |= labels[10] << 8;
		buffer.s1 |= labels[11] << 12;
		buffer.s1 |= labels[12] << 16;
		buffer.s1 |= labels[13] << 20;
		buffer.s1 |= labels[14] << 24;
		buffer.s1 |= labels[15] << 28;
		
		buffer.s2  = labels[16];
		buffer.s2 |= labels[17] << 4;
		buffer.s2 |= labels[18] << 8;
		buffer.s2 |= labels[19] << 12;
		buffer.s2 |= labels[20] << 16;
		buffer.s2 |= labels[21] << 20;
		buffer.s2 |= labels[22] << 24;
		buffer.s2 |= labels[23] << 28;

		buffer.s3  = labels[24];
		buffer.s3 |= labels[25] << 4;
		buffer.s3 |= labels[26] << 8;
		buffer.s3 |= labels[27] << 12;
		buffer.s3 |= labels[28] << 16;
		buffer.s3 |= labels[29] << 20;
		buffer.s3 |= labels[30] << 24;
		buffer.s3 |= labels[31] << 28;

		buffer.s4  = labels[32];
		buffer.s4 |= labels[33] << 4;
		buffer.s4 |= labels[34] << 8;
		buffer.s4 |= labels[35] << 12;
		buffer.s4 |= labels[36] << 16;
		buffer.s4 |= labels[37] << 20;
		buffer.s4 |= labels[38] << 24;
		buffer.s4 |= labels[39] << 28;

		buffer.s5  = labels[40];
		buffer.s5 |= labels[41] << 4;
		buffer.s5 |= labels[42] << 8;
		buffer.s5 |= labels[43] << 12;
		buffer.s5 |= labels[44] << 16;
		buffer.s5 |= labels[45] << 20;
		buffer.s5 |= labels[46] << 24;
		buffer.s5 |= labels[47] << 28;

		buffer.s6  = labels[48];
		buffer.s6 |= labels[49] << 4;
		buffer.s6 |= labels[50] << 8;
		buffer.s6 |= labels[51] << 12;
		buffer.s6 |= labels[52] << 16;
		buffer.s6 |= labels[53] << 20;
		buffer.s6 |= labels[54] << 24;
		buffer.s6 |= labels[55] << 28;

		buffer.s7  = labels[56];
		buffer.s7 |= labels[57] << 4;
		buffer.s7 |= labels[58] << 8;
		buffer.s7 |= labels[59] << 12;
		buffer.s7 |= labels[60] << 16;
		buffer.s7 |= labels[61] << 20;
		buffer.s7 |= labels[62] << 24;
		buffer.s7 |= labels[63] << 28;

		*(__global uint8*)output = buffer;
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search3(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X) & 0x07;

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint2 buffer1, buffer2, buffer3;

		output += gid * 24 / 64;

		buffer1.s0  = labels[0];
		buffer1.s0 |= labels[1] << 3;
		buffer1.s0 |= labels[2] << 6;
		buffer1.s0 |= labels[3] << 9;
		buffer1.s0 |= labels[4] << 12;
		buffer1.s0 |= labels[5] << 15;
		buffer1.s0 |= labels[6] << 18;
		buffer1.s0 |= labels[7] << 21;
		buffer1.s0 |= labels[8] << 24;
		buffer1.s0 |= labels[9] << 27;
		buffer1.s1  = labels[10];
		buffer1.s0 |= buffer1.s1 << 30;

		buffer1.s1 >>= 2;
		buffer1.s1 |= labels[11] << 1;
		buffer1.s1 |= labels[12] << 4;
		buffer1.s1 |= labels[13] << 7;
		buffer1.s1 |= labels[14] << 10;
		buffer1.s1 |= labels[15] << 13;
		buffer1.s1 |= labels[16] << 16;
		buffer1.s1 |= labels[17] << 19;
		buffer1.s1 |= labels[18] << 22;
		buffer1.s1 |= labels[19] << 25;
		buffer1.s1 |= labels[20] << 28;
		buffer2.s0  = labels[21];
		buffer1.s1 |= buffer2.s0 << 31;

		buffer2.s0 >>= 1;
		buffer2.s0 |= labels[22] << 2;
		buffer2.s0 |= labels[23] << 5;
		buffer2.s0 |= labels[24] << 8;
		buffer2.s0 |= labels[25] << 11;
		buffer2.s0 |= labels[26] << 14;
		buffer2.s0 |= labels[27] << 17;
		buffer2.s0 |= labels[28] << 20;
		buffer2.s0 |= labels[29] << 23;
		buffer2.s0 |= labels[30] << 26;
		buffer2.s0 |= labels[31] << 29;

		buffer2.s1  = labels[32+0];
		buffer2.s1 |= labels[32+1] << 3;
		buffer2.s1 |= labels[32+2] << 6;
		buffer2.s1 |= labels[32+3] << 9;
		buffer2.s1 |= labels[32+4] << 12;
		buffer2.s1 |= labels[32+5] << 15;
		buffer2.s1 |= labels[32+6] << 18;
		buffer2.s1 |= labels[32+7] << 21;
		buffer2.s1 |= labels[32+8] << 24;
		buffer2.s1 |= labels[32+9] << 27;
		buffer3.s0  = labels[32+10];
		buffer2.s1 |= buffer3.s0 << 30;

		buffer3.s0 >>= 2;
		buffer3.s0 |= labels[32+11] << 1;
		buffer3.s0 |= labels[32+12] << 4;
		buffer3.s0 |= labels[32+13] << 7;
		buffer3.s0 |= labels[32+14] << 10;
		buffer3.s0 |= labels[32+15] << 13;
		buffer3.s0 |= labels[32+16] << 16;
		buffer3.s0 |= labels[32+17] << 19;
		buffer3.s0 |= labels[32+18] << 22;
		buffer3.s0 |= labels[32+19] << 25;
		buffer3.s0 |= labels[32+20] << 28;
		buffer3.s1  = labels[32+21];
		buffer3.s0 |= buffer3.s1 << 31;

		buffer3.s1 >>= 1;
		buffer3.s1 |= labels[32+22] << 2;
		buffer3.s1 |= labels[32+23] << 5;
		buffer3.s1 |= labels[32+24] << 8;
		buffer3.s1 |= labels[32+25] << 11;
		buffer3.s1 |= labels[32+26] << 14;
		buffer3.s1 |= labels[32+27] << 17;
		buffer3.s1 |= labels[32+28] << 20;
		buffer3.s1 |= labels[32+29] << 23;
		buffer3.s1 |= labels[32+30] << 26;
		buffer3.s1 |= labels[32+31] << 29;

		((__global uint2*)output)[0] = buffer1;
		((__global uint2*)output)[1] = buffer2;
		((__global uint2*)output)[2] = buffer3;
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search2(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X) & 0x03;

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint4 buffer;

		output += gid * 16 / 64;

		buffer.s0  = labels[0];
		buffer.s0 |= labels[1] << 2;
		buffer.s0 |= labels[2] << 4;
		buffer.s0 |= labels[3] << 6;
		buffer.s0 |= labels[4] << 8;
		buffer.s0 |= labels[5] << 10;
		buffer.s0 |= labels[6] << 12;
		buffer.s0 |= labels[7] << 14;
		buffer.s0 |= labels[8] << 16;
		buffer.s0 |= labels[9] << 18;
		buffer.s0 |= labels[10] << 20;
		buffer.s0 |= labels[11] << 22;
		buffer.s0 |= labels[12] << 24;
		buffer.s0 |= labels[13] << 26;
		buffer.s0 |= labels[14] << 28;
		buffer.s0 |= labels[15] << 30;
		
		buffer.s1  = labels[16];
		buffer.s1 |= labels[17] << 2;
		buffer.s1 |= labels[18] << 4;
		buffer.s1 |= labels[19] << 6;
		buffer.s1 |= labels[20] << 8;
		buffer.s1 |= labels[21] << 10;
		buffer.s1 |= labels[22] << 12;
		buffer.s1 |= labels[23] << 14;
		buffer.s1 |= labels[24] << 16;
		buffer.s1 |= labels[25] << 18;
		buffer.s1 |= labels[26] << 20;
		buffer.s1 |= labels[27] << 22;
		buffer.s1 |= labels[28] << 24;
		buffer.s1 |= labels[29] << 26;
		buffer.s1 |= labels[30] << 28;
		buffer.s1 |= labels[31] << 30;

		buffer.s2  = labels[32];
		buffer.s2 |= labels[33] << 2;
		buffer.s2 |= labels[34] << 4;
		buffer.s2 |= labels[35] << 6;
		buffer.s2 |= labels[36] << 8;
		buffer.s2 |= labels[37] << 10;
		buffer.s2 |= labels[38] << 12;
		buffer.s2 |= labels[39] << 14;
		buffer.s2 |= labels[40] << 16;
		buffer.s2 |= labels[41] << 18;
		buffer.s2 |= labels[42] << 20;
		buffer.s2 |= labels[43] << 22;
		buffer.s2 |= labels[44] << 24;
		buffer.s2 |= labels[45] << 26;
		buffer.s2 |= labels[46] << 28;
		buffer.s2 |= labels[47] << 30;

		buffer.s3  = labels[48];
		buffer.s3 |= labels[49] << 2;
		buffer.s3 |= labels[50] << 4;
		buffer.s3 |= labels[51] << 6;
		buffer.s3 |= labels[52] << 8;
		buffer.s3 |= labels[53] << 10;
		buffer.s3 |= labels[54] << 12;
		buffer.s3 |= labels[55] << 14;
		buffer.s3 |= labels[56] << 16;
		buffer.s3 |= labels[57] << 18;
		buffer.s3 |= labels[58] << 20;
		buffer.s3 |= labels[59] << 22;
		buffer.s3 |= labels[60] << 24;
		buffer.s3 |= labels[61] << 26;
		buffer.s3 |= labels[62] << 28;
		buffer.s3 |= labels[63] << 30;

		*(__global uint4*)output = buffer;
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void search1(__global const uint4 * restrict input, volatile __global uchar * restrict output, __global uchar * restrict padcache, const uint N, const ulong nonceBase, const uint r, const uint p)
{
	__local uint labels[64];
	uint4 password[5];
	uint4 X[8];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	uint Nfactor = 0;
	uint tmp = N >> 1;
	ulong nonce = nonceBase + gid;
	
	/* Determine the Nfactor */
	while ((tmp & 1) == 0) {
		tmp >>= 1;
		Nfactor++;
	}
	
	password[0] = input[0];
	password[1] = input[1];
	password[2] = input[2];
	password[3] = input[3];
	password[4] = input[4];
	password[2].xy = as_uint2(nonce);
	
	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_128B(password, X);

	/* 2: X = ROMix(X) */
	scrypt_ROMix(X, (__global uint4 *)padcache, N, gid, Nfactor);

	/* 3: Out = PBKDF2(password, X) */
	labels[lid] = scrypt_pbkdf2_32B(password, X) & 0x01;

	barrier(CLK_LOCAL_MEM_FENCE); 

	if (0 == lid) {
		uint2 buffer;

		output += gid * 8 / 64;

		buffer.s0  = labels[0];
		buffer.s0 |= labels[1] << 1;
		buffer.s0 |= labels[2] << 2;
		buffer.s0 |= labels[3] << 3;
		buffer.s0 |= labels[4] << 4;
		buffer.s0 |= labels[5] << 5;
		buffer.s0 |= labels[6] << 6;
		buffer.s0 |= labels[7] << 7;
		buffer.s0 |= labels[8] << 8;
		buffer.s0 |= labels[9] << 9;
		buffer.s0 |= labels[10] << 10;
		buffer.s0 |= labels[11] << 11;
		buffer.s0 |= labels[12] << 12;
		buffer.s0 |= labels[13] << 13;
		buffer.s0 |= labels[14] << 14;
		buffer.s0 |= labels[15] << 15;
		buffer.s0 |= labels[16] << 16;
		buffer.s0 |= labels[17] << 17;
		buffer.s0 |= labels[18] << 18;
		buffer.s0 |= labels[19] << 19;
		buffer.s0 |= labels[20] << 20;
		buffer.s0 |= labels[21] << 21;
		buffer.s0 |= labels[22] << 22;
		buffer.s0 |= labels[23] << 23;
		buffer.s0 |= labels[24] << 24;
		buffer.s0 |= labels[25] << 25;
		buffer.s0 |= labels[26] << 26;
		buffer.s0 |= labels[27] << 27;
		buffer.s0 |= labels[28] << 28;
		buffer.s0 |= labels[29] << 29;
		buffer.s0 |= labels[30] << 30;
		buffer.s0 |= labels[31] << 31;

		buffer.s1  = labels[32];
		buffer.s1 |= labels[33] << 1;
		buffer.s1 |= labels[34] << 2;
		buffer.s1 |= labels[35] << 3;
		buffer.s1 |= labels[36] << 4;
		buffer.s1 |= labels[37] << 5;
		buffer.s1 |= labels[38] << 6;
		buffer.s1 |= labels[39] << 7;
		buffer.s1 |= labels[40] << 8;
		buffer.s1 |= labels[41] << 9;
		buffer.s1 |= labels[42] << 10;
		buffer.s1 |= labels[43] << 11;
		buffer.s1 |= labels[44] << 12;
		buffer.s1 |= labels[45] << 13;
		buffer.s1 |= labels[46] << 14;
		buffer.s1 |= labels[47] << 15;
		buffer.s1 |= labels[48] << 16;
		buffer.s1 |= labels[49] << 17;
		buffer.s1 |= labels[50] << 18;
		buffer.s1 |= labels[51] << 19;
		buffer.s1 |= labels[52] << 20;
		buffer.s1 |= labels[53] << 21;
		buffer.s1 |= labels[54] << 22;
		buffer.s1 |= labels[55] << 23;
		buffer.s1 |= labels[56] << 24;
		buffer.s1 |= labels[57] << 25;
		buffer.s1 |= labels[58] << 26;
		buffer.s1 |= labels[59] << 27;
		buffer.s1 |= labels[60] << 28;
		buffer.s1 |= labels[61] << 29;
		buffer.s1 |= labels[62] << 30;
		buffer.s1 |= labels[63] << 31;

		*(__global uint2*)output = buffer;
	}
}
