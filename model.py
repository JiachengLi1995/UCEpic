import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers import RobertaForMaskedLM, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.file_utils import ModelOutput

@dataclass
class UCEpicOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    ins_loss: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None
    ins_logits: Optional[torch.FloatTensor] = None
    lm_correct: Optional[int] = None
    lm_total: Optional[int] = None
    ins_pred: Optional[List] = None
    ins_true: Optional[List] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class UCEpic(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.aspect_embedding = nn.Embedding(config.num_aspects, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def roberta_embedding(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        past_key_values=None
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.roberta.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.roberta.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        return embedding_output

    def roberta_encoding(
        self,
        embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        input_shape = embeddings.size()[:-1]
        batch_size, seq_length = input_shape
        device = embeddings.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        extended_attention_mask: torch.Tensor = self.roberta.get_extended_attention_mask(attention_mask, input_shape, device=device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.roberta.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.roberta.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.roberta.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

    def lm_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        ref=None,  #(batch, seq_len, hidden_size)
        ref_mask=None, #(batch, seq_len)
        aspects=None, #(batch, aspect_num)
        aspects_mask=None, #(batch, aspect_num)
        lm_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeddings = self.roberta_embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        aspect_embeddings = self.roberta_embedding(inputs_embeds=self.aspect_embedding(aspects))
        ref_embeddings = self.roberta_embedding(input_ids=ref)

        aspect_length = aspect_embeddings.size(1)
        ref_length = ref_embeddings.size(1)

        embeddings = torch.cat([ref_embeddings, aspect_embeddings, input_embeddings], dim=1)
        extended_attention_mask = torch.cat([ref_mask, aspects_mask, attention_mask], dim=1)

        outputs = self.roberta_encoding(
            embeddings=embeddings,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, ref_length+aspect_length:]

        assert sequence_output.size(1)==input_ids.size(1)

        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        correct = 0
        total = 0
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))

            masked_lm_mask = lm_labels != -100
            masked_lm_label = torch.masked_select(lm_labels, masked_lm_mask)
            masked_prediction_scores = torch.masked_select(prediction_scores, masked_lm_mask.unsqueeze(-1)).view(-1, self.config.vocab_size)
            if masked_prediction_scores.size(0) > 0:
                correct = (torch.argmax(masked_prediction_scores, 1) == masked_lm_label).sum().detach().cpu().item()
                total = masked_lm_label.ne(-100).sum().detach().cpu().item()


        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return UCEpicOutput(
            masked_lm_loss=masked_lm_loss,
            lm_logits=prediction_scores,
            lm_correct=correct,
            lm_total=total,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def ins_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        ref=None,  #(batch, seq_len, hidden_size)
        ref_mask=None, #(batch, seq_len)
        aspects=None,
        aspects_mask=None,
        ins_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeddings = self.roberta_embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        aspect_embeddings = self.roberta_embedding(inputs_embeds=self.aspect_embedding(aspects))
        ref_embeddings = self.roberta_embedding(input_ids=ref)

        aspect_length = aspect_embeddings.size(1)
        ref_length = ref_embeddings.size(1)

        embeddings = torch.cat([ref_embeddings, aspect_embeddings, input_embeddings], dim=1)
        extended_attention_mask = torch.cat([ref_mask, aspects_mask, attention_mask], dim=1)

        outputs = self.roberta_encoding(
            embeddings=embeddings,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, ref_length+aspect_length:]

        assert sequence_output.size(1)==input_ids.size(1)
                                                            
        sequence_output = self.dropout(sequence_output)
        ins_scores = self.classifier(sequence_output)

        ins_loss = None
        ins_pred = []
        ins_true = []
        if ins_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ins_loss = loss_fct(ins_scores.view(-1, self.config.num_labels), ins_labels.view(-1))

            ins_mask = ins_labels != -100
            masked_ins_labels = torch.masked_select(ins_labels, ins_mask).view(-1)
            masked_ins_scores = torch.masked_select(ins_scores, ins_mask.unsqueeze(-1)).view(-1, self.config.num_labels)

            ins_pred = torch.argmax(masked_ins_scores, dim=1).detach().long().cpu().tolist()
            ins_true = masked_ins_labels.detach().cpu().tolist()

        if not return_dict:
            output = (ins_scores,) + outputs[2:]
            return ((ins_loss,) + output) if ins_loss is not None else output

        return UCEpicOutput(
            ins_loss=ins_loss,
            ins_logits=ins_scores,
            ins_pred=ins_pred,
            ins_true=ins_true,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def forward(
        self,
        input_ids_s=None,
        attention_mask_s=None,
        token_type_ids_s=None,
        position_ids_s=None,
        input_ids_m=None,
        attention_mask_m=None,
        token_type_ids_m=None,
        position_ids_m=None,
        lm_labels=None,
        ins_labels=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ref=None,  #(batch, seq_len, hidden_size)
        ref_mask=None, #(batch, seq_len)
        aspects=None, #(batch, aspect_num)
        aspects_mask=None,
    ):


        ins_outputs = self.ins_forward(
                        input_ids=input_ids_s,
                        attention_mask=attention_mask_s,
                        token_type_ids=token_type_ids_s,
                        position_ids=position_ids_s,
                        inputs_embeds=inputs_embeds,
                        ref=ref,
                        ref_mask=ref_mask,
                        aspects=aspects,
                        aspects_mask=aspects_mask,
                        ins_labels=ins_labels,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

        lm_outputs = self.lm_forward(
                        input_ids=input_ids_m,
                        attention_mask=attention_mask_m,
                        token_type_ids=token_type_ids_m,
                        position_ids=position_ids_m,
                        inputs_embeds=inputs_embeds,
                        ref=ref,
                        ref_mask=ref_mask,
                        aspects=aspects,
                        aspects_mask=aspects_mask,
                        lm_labels=lm_labels,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

        loss = None
        if ins_outputs.ins_loss is not None:
            loss = ins_outputs.ins_loss

        if lm_outputs.masked_lm_loss is not None:
            if loss is not None:
                loss += lm_outputs.masked_lm_loss
            else:
                loss = lm_outputs.masked_lm_loss

        return UCEpicOutput(
            loss=loss,
            masked_lm_loss=lm_outputs.masked_lm_loss,
            ins_loss=ins_outputs.ins_loss,
            lm_logits=lm_outputs.lm_logits,
            ins_logits=ins_outputs.ins_logits,
            lm_correct=lm_outputs.lm_correct,
            lm_total=lm_outputs.lm_total,
            ins_pred=ins_outputs.ins_pred,
            ins_true=ins_outputs.ins_true,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
        )