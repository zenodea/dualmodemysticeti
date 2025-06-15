// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::committee::Committee;
use crate::{
    consensus::{
        universal_committer::UniversalCommitterBuilder, LeaderStatus, DEFAULT_WAVE_LENGTH,
        DEFAULT_WAVE_LENGTH_ASYNC,
    },
    test_util::{build_dag, build_dag_layer, committee, test_metrics, TestBlockWriter},
    types::{BlockReference, StatementBlock},
};

/// Commit one leader.
#[test]
#[tracing_test::traced_test]
fn direct_commit() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut block_writer = TestBlockWriter::new(&committee);
    build_dag(&committee, &mut block_writer, None, wave_length);

    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");

    assert_eq!(sequence.len(), 1);
    if let LeaderStatus::Commit(ref block) = sequence[0] {
        assert_eq!(block.author(), committee.elect_leader(1));
    } else {
        panic!("Expected a committed leader")
    };
}

/// Ensure idempotent replies.
#[test]
#[tracing_test::traced_test]
fn idempotence() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut block_writer = TestBlockWriter::new(&committee);
    build_dag(&committee, &mut block_writer, None, 5);

    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    // Commit one block.
    let last_committed = BlockReference::new_test(0, 0);
    let committed = committer.try_commit(last_committed);

    // Ensure we don't commit it again.
    let last = committed.into_iter().last().unwrap();
    let last_committed = BlockReference::new_test(last.authority(), last.round());
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");
    assert!(sequence.is_empty());
}

/// Commit one by one each leader as the dag progresses in ideal conditions.
#[test]
#[tracing_test::traced_test]
fn multiple_direct_commit() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut last_committed = BlockReference::new_test(0, 0);
    for n in 1..=10 {
        let enough_blocks = n + (wave_length - 1);
        let mut block_writer = TestBlockWriter::new(&committee);
        build_dag(&committee, &mut block_writer, None, enough_blocks);

        let committer = UniversalCommitterBuilder::new(
            committee.clone(),
            block_writer.into_block_store(),
            test_metrics(),
        )
        .with_wave_length(wave_length)
        .with_pipeline(true)
        .build();

        let sequence = committer.try_commit(last_committed);
        tracing::info!("Commit sequence: {sequence:?}");

        assert_eq!(sequence.len(), 1);
        let leader_round = n as u64;
        if let LeaderStatus::Commit(ref block) = sequence[0] {
            assert_eq!(block.author(), committee.elect_leader(leader_round));
        } else {
            panic!("Expected a committed leader")
        }

        let last = sequence.into_iter().last().unwrap();
        last_committed = BlockReference::new_test(last.authority(), last.round());
    }
}

/// Commit 10 leaders in a row (calling the committer after adding them).
#[test]
#[tracing_test::traced_test]
fn direct_commit_late_call() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let n = 10;
    let enough_blocks = n + (wave_length - 1);
    let mut block_writer = TestBlockWriter::new(&committee);
    build_dag(&committee, &mut block_writer, None, enough_blocks);

    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");

    assert_eq!(sequence.len(), n as usize);
    for (i, leader_block) in sequence.iter().enumerate() {
        let leader_round = 1 + i as u64;
        if let LeaderStatus::Commit(ref block) = leader_block {
            assert_eq!(block.author(), committee.elect_leader(leader_round));
        } else {
            panic!("Expected a committed leader")
        };
    }
}

/// Do not commit anything if we are still in the first wave.
#[test]
#[tracing_test::traced_test]
fn no_genesis_commit() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let first_commit_round = wave_length - 1;
    for r in 0..first_commit_round {
        let mut block_writer = TestBlockWriter::new(&committee);
        build_dag(&committee, &mut block_writer, None, r);

        let committer = UniversalCommitterBuilder::new(
            committee.clone(),
            block_writer.into_block_store(),
            test_metrics(),
        )
        .with_wave_length(wave_length)
        .with_pipeline(true)
        .build();

        let last_committed = BlockReference::new_test(0, 0);
        let sequence = committer.try_commit(last_committed);
        tracing::info!("Commit sequence: {sequence:?}");
        assert!(sequence.is_empty());
    }
}

/// We do not commit anything if we miss the first leader.
#[test]
#[tracing_test::traced_test]
fn no_leader() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut block_writer = TestBlockWriter::new(&committee);

    // Add enough blocks to reach the decision round of the first leader (but without the leader).
    let leader_round_1 = 1;
    let leader_1 = committee.elect_leader(leader_round_1);

    let genesis: Vec<_> = committee
        .authorities()
        .map(|authority| *StatementBlock::new_genesis(authority).reference())
        .collect();
    let connections = committee
        .authorities()
        .filter(|&authority| authority != leader_1)
        .map(|authority| (authority, genesis.clone()));
    let references = build_dag_layer(connections.collect(), &mut block_writer);

    let decision_round_1 = wave_length;
    build_dag(
        &committee,
        &mut block_writer,
        Some(references),
        decision_round_1,
    );

    // Ensure no blocks are committed.
    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");

    assert_eq!(sequence.len(), 1);
    if let LeaderStatus::Skip(leader, round) = sequence[0] {
        assert_eq!(leader, leader_1);
        assert_eq!(round, leader_round_1);
    } else {
        panic!("Expected to directly skip the leader");
    }
}

/// We directly skip the leader if it has enough blame.
#[test]
#[tracing_test::traced_test]
fn direct_skip() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut block_writer = TestBlockWriter::new(&committee);

    // Add enough blocks to reach the first leader.
    let leader_round_1 = 1;
    let references_1 = build_dag(&committee, &mut block_writer, None, leader_round_1);

    // Filter out that leader.
    let references_without_leader_1: Vec<_> = references_1
        .into_iter()
        .filter(|x| x.authority != committee.elect_leader(leader_round_1))
        .collect();

    // Add enough blocks to reach the decision round of the first leader.
    let decision_round_1 = wave_length;
    build_dag(
        &committee,
        &mut block_writer,
        Some(references_without_leader_1),
        decision_round_1,
    );

    // Ensure the omitted leader is skipped.
    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");

    assert_eq!(sequence.len(), 1);
    if let LeaderStatus::Skip(leader, round) = sequence[0] {
        assert_eq!(leader, committee.elect_leader(leader_round_1));
        assert_eq!(round, leader_round_1);
    } else {
        panic!("Expected to directly skip the leader");
    }
}

/// Indirect-commit the first leader.
#[test]
#[tracing_test::traced_test]
fn indirect_commit() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut block_writer = TestBlockWriter::new(&committee);

    // Add enough blocks to reach the 1st leader.
    let leader_round_1 = 1;
    let references_1 = build_dag(&committee, &mut block_writer, None, leader_round_1);

    // Filter out that leader.
    let references_without_leader_1: Vec<_> = references_1
        .iter()
        .cloned()
        .filter(|x| x.authority != committee.elect_leader(leader_round_1))
        .collect();

    // Only 2f+1 validators vote for the 1st leader.
    let connections_with_leader_1 = committee
        .authorities()
        .take(committee.quorum_threshold() as usize)
        .map(|authority| (authority, references_1.clone()))
        .collect();
    let references_with_votes_for_leader_1 =
        build_dag_layer(connections_with_leader_1, &mut block_writer);

    let connections_without_leader_1 = committee
        .authorities()
        .skip(committee.quorum_threshold() as usize)
        .map(|authority| (authority, references_without_leader_1.clone()))
        .collect();
    let references_without_votes_for_leader_1 =
        build_dag_layer(connections_without_leader_1, &mut block_writer);

    // Only f+1 validators certify the 1st leader.
    let mut references_3 = Vec::new();

    let connections_with_votes_for_leader_1 = committee
        .authorities()
        .take(committee.validity_threshold() as usize)
        .map(|authority| (authority, references_with_votes_for_leader_1.clone()))
        .collect();
    references_3.extend(build_dag_layer(
        connections_with_votes_for_leader_1,
        &mut block_writer,
    ));

    let references: Vec<_> = references_without_votes_for_leader_1
        .into_iter()
        .chain(references_with_votes_for_leader_1.into_iter())
        .take(committee.quorum_threshold() as usize)
        .collect();
    let connections_without_votes_for_leader_1 = committee
        .authorities()
        .skip(committee.validity_threshold() as usize)
        .map(|authority| (authority, references.clone()))
        .collect();
    references_3.extend(build_dag_layer(
        connections_without_votes_for_leader_1,
        &mut block_writer,
    ));

    // Add enough blocks to decide the 5th leader. The second leader may be skipped
    // (if it was the vote for the first leader that we removed) so we add enough blocks
    // to recursively decide it.
    let decision_round_3 = 2 * wave_length + 1;
    build_dag(
        &committee,
        &mut block_writer,
        Some(references_3),
        decision_round_3,
    );

    // Ensure we commit the first leaders.
    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");
    assert_eq!(sequence.len(), 5);

    let leader_round = 1;
    let leader = committee.elect_leader(leader_round);
    if let LeaderStatus::Commit(ref block) = sequence[0] {
        assert_eq!(block.author(), leader);
    } else {
        panic!("Expected a committed leader")
    };
}

/// Commit the first 3 leaders, skip the 4th, and commit the next 3 leaders.
#[test]
#[tracing_test::traced_test]
fn indirect_skip() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut block_writer = TestBlockWriter::new(&committee);

    // Add enough blocks to reach the 4th leader.
    let leader_round_4 = wave_length + 1;
    let references_4 = build_dag(&committee, &mut block_writer, None, leader_round_4);

    // Filter out that leader.
    let references_without_leader_4: Vec<_> = references_4
        .iter()
        .cloned()
        .filter(|x| x.authority != committee.elect_leader(leader_round_4))
        .collect();

    // Only f+1 validators connect to the 4th leader.
    let mut references_5 = Vec::new();

    let connections_with_leader_4 = committee
        .authorities()
        .take(committee.validity_threshold() as usize)
        .map(|authority| (authority, references_4.clone()))
        .collect();
    references_5.extend(build_dag_layer(
        connections_with_leader_4,
        &mut block_writer,
    ));

    let connections_without_leader_4 = committee
        .authorities()
        .skip(committee.validity_threshold() as usize)
        .map(|authority| (authority, references_without_leader_4.clone()))
        .collect();
    references_5.extend(build_dag_layer(
        connections_without_leader_4,
        &mut block_writer,
    ));

    // Add enough blocks to reach the decision round of the 7th leader.
    let decision_round_7 = 3 * wave_length;
    build_dag(
        &committee,
        &mut block_writer,
        Some(references_5),
        decision_round_7,
    );

    // Ensure we commit the first 3 leaders, skip the 4th, and commit the last 2 leaders.
    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");
    assert_eq!(sequence.len(), 7);

    // Ensure we commit the first 3 leaders.
    for i in 0..=2 {
        let leader_round = i + 1;
        let leader = committee.elect_leader(leader_round);
        if let LeaderStatus::Commit(ref block) = sequence[i as usize] {
            assert_eq!(block.author(), leader);
        } else {
            panic!("Expected a committed leader")
        };
    }

    // Ensure we skip the leader of wave 1 (first pipeline) but commit the others.
    if let LeaderStatus::Skip(leader, round) = sequence[3] {
        assert_eq!(leader, committee.elect_leader(leader_round_4));
        assert_eq!(round, leader_round_4);
    } else {
        panic!("Expected a skipped leader")
    }

    for i in 4..=6 {
        let leader_round = i + 1;
        let leader = committee.elect_leader(leader_round);
        if let LeaderStatus::Commit(ref block) = sequence[i as usize] {
            assert_eq!(block.author(), leader);
        } else {
            panic!("Expected a committed leader")
        };
    }
}

/// If there is no leader with enough support nor blame, we commit nothing.
#[test]
#[tracing_test::traced_test]
fn undecided() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;

    let mut block_writer = TestBlockWriter::new(&committee);

    // Add enough blocks to reach the first leader.
    let leader_round_1 = 1;
    let references_1 = build_dag(&committee, &mut block_writer, None, leader_round_1);

    // Filter out that leader.
    let references_1_without_leader: Vec<_> = references_1
        .iter()
        .cloned()
        .filter(|x| x.authority != committee.elect_leader(leader_round_1))
        .collect();

    // Create a dag layer where only one authority votes for the first leader.
    let mut authorities = committee.authorities();
    let leader_connection = vec![(authorities.next().unwrap(), references_1)];
    let non_leader_connections: Vec<_> = authorities
        .take((committee.quorum_threshold() - 1) as usize)
        .map(|authority| (authority, references_1_without_leader.clone()))
        .collect();

    let connections = leader_connection.into_iter().chain(non_leader_connections);
    let references = build_dag_layer(connections.collect(), &mut block_writer);

    // Add enough blocks to reach the first decision round
    let decision_round_1 = wave_length;
    build_dag(
        &committee,
        &mut block_writer,
        Some(references),
        decision_round_1,
    );

    // Ensure no blocks are committed.
    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");
    assert!(sequence.is_empty());
}

///Dual-mode Test
///Multi-Committer Version

// Direct to-commit in async mode
#[test]
#[tracing_test::traced_test]
fn direct_commit_switch_round() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;
    let wave_length_async = DEFAULT_WAVE_LENGTH_ASYNC;

    // switch round begins after one full wave of mysticeti is completed
    let switch_round = wave_length;
    let decision_round = switch_round + wave_length_async - 1;

    // Add enough blocks to reach the first leader.
    let mut block_writer = TestBlockWriter::new(&committee);
    build_dag(&committee, &mut block_writer, None, decision_round);

    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_async_wave_length(wave_length_async)
    .with_switch_round(switch_round)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");

    // Sequence length will be 5, first 3 rounds to reach wave will always committ a leader
    // switch_round will committ a leader, alongside the first boost round of the asynchronous wave
    assert_eq!(sequence.len(), 5);
    if let LeaderStatus::Commit(ref block) = sequence[0] {
        assert_eq!(block.author(), committee.elect_leader(1));
    } else {
        panic!("Expected a committed leader")
    };
}

// direct to-skip switch round Working
#[test]
#[tracing_test::traced_test]
fn direct_skip_switch_round() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;
    let wave_length_async = DEFAULT_WAVE_LENGTH_ASYNC;

    let switch_round = wave_length;

    // Add enough blocks to reach leader of switch round.
    let mut block_writer = TestBlockWriter::new(&committee);
    let references_1 = build_dag(&committee, &mut block_writer, None, switch_round);

    // Filter out leader of switch round.
    let references_without_leader_1: Vec<_> = references_1
        .into_iter()
        .filter(|x| x.authority != committee.elect_leader(switch_round))
        .collect();

    // Add enough blocks to reach the decision round of the switch round leader
    let decision_round_1 = wave_length * 1 + wave_length_async;
    build_dag(
        &committee,
        &mut block_writer,
        Some(references_without_leader_1),
        decision_round_1,
    );

    // Ensure the leader of switch round is skipped.
    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_async_wave_length(wave_length_async)
    .with_switch_round(switch_round)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");

    // Sequence length will be 5, first 3 rounds to reach wave will always committ a leader
    // switch_round will committ a leader, alongside the first boost round of the asynchronous wave
    assert_eq!(sequence.len(), 5);
    if let LeaderStatus::Skip(leader, round) = sequence[0] {
        assert_eq!(leader, committee.elect_leader(switch_round));
        assert_eq!(round, switch_round);
    } else {
        panic!("Expected to directly skip the leader");
    }
}

// direct-undecided conclusion for switch round (works for both 4 and 5 wave length for
// the asynchronous wave)
#[test]
#[tracing_test::traced_test]
fn undecided_switch_round() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;
    let wave_length_async = DEFAULT_WAVE_LENGTH_ASYNC;
    let switch_round = wave_length;

    let mut block_writer = TestBlockWriter::new(&committee);

    let base_references = build_dag(&committee, &mut block_writer, None, switch_round);

    // Build dag where async wave leader is undecided,
    // extended (wave_length*2) enough to make it so that the leader is skipped via the rule
    // Modifies block_writer to contain the necessary dag formation to reach the desired conclusion
    // (undecided conclusion)
    simulate_undecided_switch_round(
        &base_references,
        &committee,
        &mut block_writer,
        switch_round,
        wave_length_async,
    );

    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_async_wave_length(wave_length_async)
    .with_switch_round(switch_round)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");
    assert!(sequence.is_empty());
}

// Indirect-commit the leader of the switch_round (asynchronous wave)
#[test]
#[tracing_test::traced_test]
fn indirect_commit_switch_round() {
    let committee = committee(4);
    let wave_length = DEFAULT_WAVE_LENGTH;
    let wave_length_async = DEFAULT_WAVE_LENGTH_ASYNC;
    let switch_round = wave_length;

    let mut block_writer = TestBlockWriter::new(&committee);

    let base_refereneces = build_dag(&committee, &mut block_writer, None, switch_round);

    // Build dag to reach undecided verdict for leader round
    let indirect_skip_references = simulate_undecided_switch_round(
        &base_refereneces,
        &committee,
        &mut block_writer,
        switch_round,
        wave_length_async,
    );

    // reach decision_round for the first possible anchor block
    // first mysticeti wave is not enough to reach the block
    let decision_round = switch_round + 2 * wave_length_async;

    build_dag(
        &committee,
        &mut block_writer,
        Some(indirect_skip_references),
        decision_round,
    );

    let committer = UniversalCommitterBuilder::new(
        committee.clone(),
        block_writer.into_block_store(),
        test_metrics(),
    )
    .with_wave_length(wave_length)
    .with_async_wave_length(wave_length_async)
    .with_switch_round(switch_round)
    .with_pipeline(true)
    .build();

    let last_committed = BlockReference::new_test(0, 0);
    let sequence = committer.try_commit(last_committed);
    tracing::info!("Commit sequence: {sequence:?}");
    // 11 instead of 5 due to the required extra rounds
    // to reach round threhsold for mahi-mahi's anchor block
    assert_eq!(sequence.len(), 11);

    let leader_round = 1;
    let leader = committee.elect_leader(leader_round);
    if let LeaderStatus::Commit(ref block) = sequence[0] {
        assert_eq!(block.author(), leader);
    } else {
        panic!("Expected a committed leader")
    };
}

// Function to remove redundant code
// returns dag structure where proposed leader block does not have enough blames
// or enough votes for the direct rule to committ/skip the block, takes into account
// boost rounds.
fn simulate_undecided_switch_round(
    base_reference: &Vec<BlockReference>,
    committee: &Committee,
    block_writer: &mut TestBlockWriter,
    switch_round: u64,
    wave_length_async: u64,
) -> Vec<BlockReference> {
    let mut reference = base_reference.clone();
    // Identify leader for switch round (leader of asynchronous wave)
    let leader = committee.elect_leader(switch_round);

    // Structure follows DAG shown in figure 6, example c of the Mahi-Mahi research paper
    for i in 0..wave_length_async {
        // Store [A,B,C,D] block references
        let vote_for_leader = reference.clone(); // Choice A: Include leader

        // Store lock references without leader block
        let vote_against_leader: Vec<_> = reference
            .iter()
            .cloned()
            .filter(|x| x.authority != leader)
            .collect(); // Choice B: Exclude leader

        // NOTE: I don't like this, I should probably find a cleaner solution
        let connections = if i != wave_length_async - 3 {
            // If boost round, or certificate round, only the leader in question will contain
            // the block desired
            // A, B, C blocks : [A, B, C]
            // D block : [A, B, C, D]
            committee
                .authorities()
                .map(|authority| {
                    if authority == leader {
                        (authority, vote_for_leader.clone())
                    } else {
                        (authority, vote_against_leader.clone())
                    }
                })
                .collect()
        } else {
            // If vote round, the leader who proposed the block, alongside 2f other validators
            // Will include a proposed block that references the block in question
            // A blocks : [A, B, C]
            // B, C, D blocks : [A, B, C, D]
            committee
                .authorities()
                .filter(|&authority| authority == leader)
                .take(1)
                .map(|authority| (authority, vote_for_leader.clone()))
                .chain(
                    committee
                        .authorities()
                        .filter(|&authority| authority != leader)
                        .take((committee.quorum_threshold() - 1) as usize)
                        .map(|authority| (authority, vote_for_leader.clone())),
                )
                .chain(
                    committee
                        .authorities()
                        .filter(|&authority| authority != leader)
                        .skip((committee.validity_threshold()) as usize)
                        .map(|authority| (authority, vote_against_leader.clone())),
                )
                .collect()
        };
        reference = build_dag_layer(connections, block_writer);
    }
    reference
}
