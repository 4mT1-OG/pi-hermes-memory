import { describe, it } from 'node:test';
import * as assert from 'node:assert/strict';
import {
  DEFAULT_BOUNDED_RETRIEVAL_TOKEN_CAP,
  assembleRetrievalPrompt,
  countFinalPromptTokens,
  createNoSafeResult,
  type Tokenizer,
} from '../src/bounded-retrieval.js';

describe('bounded retrieval scaffold', () => {
  it('uses the configured 2,500-token retrieval cap', () => {
    assert.strictEqual(DEFAULT_BOUNDED_RETRIEVAL_TOKEN_CAP, 2500);
  });

  it('counts the final assembled retrieval prompt through the tokenizer', async () => {
    const tokenizer: Tokenizer = { countTokens: (text: string) => text.length };
    const prompt = assembleRetrievalPrompt(['label: auth', 'auth rotation policy']);

    assert.strictEqual(prompt, 'label: auth\n\nauth rotation policy');
    assert.strictEqual(await countFinalPromptTokens(prompt, tokenizer), prompt.length);
  });

  it('returns a no-safe-result decision without injecting', () => {
    const decision = createNoSafeResult('no_safe_result');

    assert.strictEqual(decision.injected, false);
    assert.strictEqual(decision.reason, 'no_safe_result');
  });
});
