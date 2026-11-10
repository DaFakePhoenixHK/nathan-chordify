import { describe, expect, it } from 'vitest';
import { isDownStatus, parsePair } from './connection';

const CODE = 'abcdefghijklmnopqrstuvwx';
const LINK = `https://dafakephoenixhk.github.io/nathan-chordify/#/pair?s=${encodeURIComponent('https://a-b.example.com')}&t=${CODE}`;

describe('parsePair', () => {
  it('reads a full pairing link', () => {
    expect(parsePair(LINK)).toEqual({ server: 'https://a-b.example.com', token: CODE });
  });
  it('reads only the part after the #', () => {
    expect(parsePair(`#/pair?s=${encodeURIComponent('https://x.example.com/')}&t=${CODE}`)).toEqual({ server: 'https://x.example.com', token: CODE });
  });
  it('without an address means this page\'s own computer', () => {
    expect(parsePair(`https://x.example.com/#/pair?t=${CODE}`)).toEqual({ server: '', token: CODE });
  });
  it('ignores spaces around a pasted link', () => {
    expect(parsePair(`  ${LINK}\n`)?.token).toBe(CODE);
  });
  it('allows a computer on this device for testing', () => {
    expect(parsePair(`#/pair?s=${encodeURIComponent('http://127.0.0.1:8769')}&t=${CODE}`)?.server).toBe('http://127.0.0.1:8769');
  });
  it('refuses unencrypted addresses', () => {
    expect(parsePair(`#/pair?s=${encodeURIComponent('http://example.com')}&t=${CODE}`)).toBeNull();
  });
  it('refuses addresses that carry a path', () => {
    expect(parsePair(`#/pair?s=${encodeURIComponent('https://example.com/x')}&t=${CODE}`)).toBeNull();
  });
  it('refuses a short or odd code', () => {
    expect(parsePair('#/pair?s=https%3A%2F%2Fa.example.com&t=short')).toBeNull();
    expect(parsePair(`#/pair?t=${CODE}<script>`)).toBeNull();
  });
  it('refuses text that is not a pairing link', () => {
    expect(parsePair('https://www.youtube.com/watch?v=abc')).toBeNull();
  });
});

describe('isDownStatus', () => {
  it('knows the computer is away', () => {
    for (const s of [0, 401, 502, 503, 504, 530]) expect(isDownStatus(s)).toBe(true);
  });
  it('treats real answers as answers', () => {
    for (const s of [200, 400, 404, 409, 500]) expect(isDownStatus(s)).toBe(false);
  });
});
