package com.microagri.app.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.microagri.app.MicroAgriApplication
import com.microagri.app.R
import com.microagri.app.databinding.ActivityHistoryBinding
import com.microagri.app.ml.DiagnosisRecord

class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val app     = application as MicroAgriApplication
        val history = if (app.modelsLoaded) app.repository.getHistory() else emptyList()

        if (history.isEmpty()) {
            binding.tvEmpty.visibility      = View.VISIBLE
            binding.recyclerView.visibility = View.GONE
        } else {
            binding.tvEmpty.visibility      = View.GONE
            binding.recyclerView.visibility = View.VISIBLE
            binding.recyclerView.layoutManager = LinearLayoutManager(this)
            binding.recyclerView.adapter = HistoryAdapter(history)
        }

        binding.btnBack.setOnClickListener { finish() }
    }
}

class HistoryAdapter(private val items: List<DiagnosisRecord>) :
    RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvDisease:    TextView = view.findViewById(R.id.tv_disease)
        val tvConfidence: TextView = view.findViewById(R.id.tv_confidence)
        val tvTimestamp:  TextView = view.findViewById(R.id.tv_timestamp)
        val tvSeverity:   TextView = view.findViewById(R.id.tv_severity)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]
        holder.tvDisease.text    = item.displayLabel
        holder.tvConfidence.text = "${(item.confidence * 100).toInt()}%"
        holder.tvTimestamp.text  = item.timestamp
        holder.tvSeverity.text   = item.severity
    }

    override fun getItemCount() = items.size
}
