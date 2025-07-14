//! Cryptographic performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neural_comm::{
    crypto::{
        CipherSuite, KeyPair,
        symmetric::{SymmetricCipher, ChaCha20Poly1305Cipher, AesGcmCipher, SymmetricKey},
        asymmetric::{AsymmetricCipher, Ed25519KeyPair, EcdsaKeyPair},
        hash::{HashFunction, Sha3Hash, Blake3Hash},
        kdf::{KeyDerivation, HkdfDerivation, Argon2Derivation},
        random::SystemRng,
    },
};

/// Benchmark symmetric encryption operations
fn bench_symmetric_crypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_crypto");
    
    // Test different data sizes
    let sizes = vec![64, 256, 1024, 4096, 16384, 65536];
    
    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // ChaCha20-Poly1305 benchmarks
        let chacha_key = SymmetricKey::generate(32).unwrap();
        let chacha_cipher = ChaCha20Poly1305Cipher::new(&chacha_key).unwrap();
        let data = vec![0u8; size];
        let aad = b"associated data";
        
        group.bench_with_input(
            BenchmarkId::new("chacha20_poly1305_encrypt", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(chacha_cipher.encrypt(black_box(&data), black_box(aad)).unwrap())
                });
            },
        );
        
        let encrypted = chacha_cipher.encrypt(&data, aad).unwrap();
        group.bench_with_input(
            BenchmarkId::new("chacha20_poly1305_decrypt", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(chacha_cipher.decrypt(black_box(&encrypted), black_box(aad)).unwrap())
                });
            },
        );
        
        // AES-GCM benchmarks
        let aes_key = SymmetricKey::generate(32).unwrap();
        let aes_cipher = AesGcmCipher::new(&aes_key).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("aes_gcm_encrypt", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(aes_cipher.encrypt(black_box(&data), black_box(aad)).unwrap())
                });
            },
        );
        
        let aes_encrypted = aes_cipher.encrypt(&data, aad).unwrap();
        group.bench_with_input(
            BenchmarkId::new("aes_gcm_decrypt", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(aes_cipher.decrypt(black_box(&aes_encrypted), black_box(aad)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark asymmetric cryptography operations
fn bench_asymmetric_crypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_crypto");
    
    // Ed25519 benchmarks
    group.bench_function("ed25519_keygen", |b| {
        b.iter(|| {
            black_box(Ed25519KeyPair::generate().unwrap())
        });
    });
    
    let ed25519_keypair = Ed25519KeyPair::generate().unwrap();
    let test_data = b"test message for signing";
    
    group.bench_function("ed25519_sign", |b| {
        b.iter(|| {
            black_box(ed25519_keypair.sign(black_box(test_data)).unwrap())
        });
    });
    
    let ed25519_signature = ed25519_keypair.sign(test_data).unwrap();
    group.bench_function("ed25519_verify", |b| {
        b.iter(|| {
            black_box(Ed25519KeyPair::verify(
                black_box(ed25519_keypair.public_key()),
                black_box(test_data),
                black_box(&ed25519_signature)
            ).unwrap())
        });
    });
    
    // ECDSA benchmarks
    group.bench_function("ecdsa_keygen", |b| {
        b.iter(|| {
            black_box(EcdsaKeyPair::generate().unwrap())
        });
    });
    
    let ecdsa_keypair = EcdsaKeyPair::generate().unwrap();
    
    group.bench_function("ecdsa_sign", |b| {
        b.iter(|| {
            black_box(ecdsa_keypair.sign(black_box(test_data)).unwrap())
        });
    });
    
    let ecdsa_signature = ecdsa_keypair.sign(test_data).unwrap();
    group.bench_function("ecdsa_verify", |b| {
        b.iter(|| {
            black_box(EcdsaKeyPair::verify(
                black_box(ecdsa_keypair.public_key()),
                black_box(test_data),
                black_box(&ecdsa_signature)
            ).unwrap())
        });
    });
    
    group.finish();
}

/// Benchmark hash functions
fn bench_hash_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_functions");
    
    let sizes = vec![64, 256, 1024, 4096, 16384, 65536];
    
    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        let data = vec![0u8; size];
        
        // SHA-3 benchmarks
        let sha3_hasher = Sha3Hash::new();
        group.bench_with_input(
            BenchmarkId::new("sha3_256", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(sha3_hasher.hash(black_box(&data)))
                });
            },
        );
        
        // BLAKE3 benchmarks
        let blake3_hasher = Blake3Hash::new();
        group.bench_with_input(
            BenchmarkId::new("blake3", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(blake3_hasher.hash(black_box(&data)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark key derivation functions
fn bench_key_derivation(c: &mut Criterion) {
    let mut group = c.benchmark_group("key_derivation");
    
    // HKDF benchmarks
    let hkdf = HkdfDerivation::new();
    let ikm = b"input key material for derivation";
    let info = b"context information";
    let mut output = [0u8; 32];
    
    group.bench_function("hkdf_derive_32", |b| {
        b.iter(|| {
            black_box(hkdf.derive(black_box(ikm), black_box(info), black_box(&mut output)).unwrap())
        });
    });
    
    let mut large_output = [0u8; 256];
    group.bench_function("hkdf_derive_256", |b| {
        b.iter(|| {
            black_box(hkdf.derive(black_box(ikm), black_box(info), black_box(&mut large_output)).unwrap())
        });
    });
    
    // Argon2 benchmarks (note: these will be slower)
    let argon2 = Argon2Derivation::new();
    let password = b"password";
    let salt = b"salt1234567890123456789012345678";
    
    group.bench_function("argon2_derive", |b| {
        b.iter(|| {
            black_box(argon2.derive_from_password(black_box(password), black_box(salt), black_box(&mut output)).unwrap())
        });
    });
    
    group.finish();
}

/// Benchmark random number generation
fn bench_random_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_generation");
    
    let mut rng = SystemRng::new().unwrap();
    
    // Different buffer sizes
    let sizes = vec![32, 64, 256, 1024, 4096];
    
    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("system_rng", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    black_box(rng.generate_bytes(black_box(size)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark complete keypair operations
fn bench_keypair_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("keypair_operations");
    
    // ChaCha20-Poly1305 keypair
    group.bench_function("chacha20_keypair_generate", |b| {
        b.iter(|| {
            black_box(KeyPair::generate(black_box(CipherSuite::ChaCha20Poly1305)).unwrap())
        });
    });
    
    let chacha_keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
    let test_data = b"neural communication test data";
    
    group.bench_function("chacha20_keypair_sign", |b| {
        b.iter(|| {
            black_box(chacha_keypair.sign(black_box(test_data)).unwrap())
        });
    });
    
    let chacha_signature = chacha_keypair.sign(test_data).unwrap();
    group.bench_function("chacha20_keypair_verify", |b| {
        b.iter(|| {
            black_box(chacha_keypair.verify(black_box(test_data), black_box(&chacha_signature)).unwrap())
        });
    });
    
    // AES-GCM keypair
    group.bench_function("aes_gcm_keypair_generate", |b| {
        b.iter(|| {
            black_box(KeyPair::generate(black_box(CipherSuite::AesGcm256)).unwrap())
        });
    });
    
    let aes_keypair = KeyPair::generate(CipherSuite::AesGcm256).unwrap();
    
    group.bench_function("aes_gcm_keypair_sign", |b| {
        b.iter(|| {
            black_box(aes_keypair.sign(black_box(test_data)).unwrap())
        });
    });
    
    let aes_signature = aes_keypair.sign(test_data).unwrap();
    group.bench_function("aes_gcm_keypair_verify", |b| {
        b.iter(|| {
            black_box(aes_keypair.verify(black_box(test_data), black_box(&aes_signature)).unwrap())
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_symmetric_crypto,
    bench_asymmetric_crypto,
    bench_hash_functions,
    bench_key_derivation,
    bench_random_generation,
    bench_keypair_operations
);
criterion_main!(benches);